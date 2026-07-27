# -*- coding: utf-8 -*-

import os
from typing import Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pgvector.sqlalchemy import Vector
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from ingredient_extractor import extract_ingredients


load_dotenv()

DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/petcheck_db",
)
CLOVA_EMBEDDING_URL = (
    "https://clovastudio.stream.ntruss.com/v1/api-tools/embedding/v2"
)
CLOVA_API_KEY = os.getenv("CLOVA_API_KEY", "")
CLOVA_REQUEST_ID = os.getenv("CLOVA_REQUEST_ID", "")

engine = create_engine(DB_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

app = FastAPI(
    title="PetCheck RAG Engine",
    description="OCR 원료를 추출하고 CLOVA Embedding v2와 pgvector로 검색합니다.",
    version="1.0.0",
)


class IngredientKnowledge(Base):
    __tablename__ = "ingredient_knowledge"

    id = Column(Integer, primary_key=True, index=True)
    ingredient_name = Column(String(100), nullable=False)
    safety_level = Column(String(20), nullable=True)
    description = Column(Text, nullable=True)
    embedding = Column(Vector(1024), nullable=True)


class RagSearchRequest(BaseModel):
    analysisId: int
    ocrText: str = Field(min_length=1)
    topK: int = Field(default=1, ge=1, le=10)
    petType: Optional[str] = "DOG"


class ContextItem(BaseModel):
    ocrIngredient: str
    ingredientName: str
    safetyLevel: Optional[str] = None
    description: Optional[str] = None
    similarityScore: float


class RagSearchResponse(BaseModel):
    analysisId: int
    extractedIngredients: list[str]
    totalCount: int
    contexts: list[ContextItem]


def get_clova_embedding(text: str) -> list[float]:
    if not CLOVA_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="CLOVA_API_KEY 환경변수가 설정되지 않았습니다.",
        )

    headers = {
        "Authorization": CLOVA_API_KEY,
        "Content-Type": "application/json",
    }
    if CLOVA_REQUEST_ID:
        headers["X-NCP-CLOVASTUDIO-REQUEST-ID"] = CLOVA_REQUEST_ID

    try:
        response = requests.post(
            CLOVA_EMBEDDING_URL,
            json={"text": text},
            headers=headers,
            timeout=10,
        )
        response.raise_for_status()
        result_data = response.json()
    except requests.RequestException as error:
        raise HTTPException(
            status_code=502,
            detail=f"Clova Embedding API 호출 실패: {error}",
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=502,
            detail="Clova Embedding API가 올바른 JSON을 반환하지 않았습니다.",
        ) from error

    if result_data.get("status", {}).get("code") != "20000":
        raise HTTPException(
            status_code=502,
            detail=f"Clova Embedding API 오류: {result_data}",
        )

    embedding = result_data.get("result", {}).get("embedding")
    if not isinstance(embedding, list) or len(embedding) != 1024:
        raise HTTPException(
            status_code=502,
            detail="Clova Embedding API의 벡터 차원이 1024가 아닙니다.",
        )

    return embedding


def deduplicate_ingredients(ingredients: list[str]) -> list[str]:
    result = []
    seen = set()

    for ingredient in ingredients:
        cleaned = ingredient.strip()
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)

    return result


@app.post("/api/v1/rag/search", response_model=RagSearchResponse)
def search_rag_context(request: RagSearchRequest) -> RagSearchResponse:
    ingredients = deduplicate_ingredients(
        extract_ingredients(request.ocrText)
    )
    if not ingredients:
        raise HTTPException(
            status_code=422,
            detail="OCR 텍스트에서 원료명을 찾지 못했습니다.",
        )

    db = SessionLocal()
    contexts = []

    try:
        for ingredient in ingredients:
            query_vector = get_clova_embedding(ingredient)
            distance = IngredientKnowledge.embedding.cosine_distance(
                query_vector
            )

            results = (
                db.query(
                    IngredientKnowledge,
                    distance.label("distance"),
                )
                .filter(IngredientKnowledge.embedding.isnot(None))
                .order_by(distance)
                .limit(request.topK)
                .all()
            )

            for item, cosine_distance in results:
                contexts.append(
                    ContextItem(
                        ocrIngredient=ingredient,
                        ingredientName=item.ingredient_name,
                        safetyLevel=item.safety_level,
                        description=item.description,
                        similarityScore=round(
                            1.0 - float(cosine_distance),
                            4,
                        ),
                    )
                )
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"RAG 검색 실행 오류: {error}",
        ) from error
    finally:
        db.close()

    return RagSearchResponse(
        analysisId=request.analysisId,
        extractedIngredients=ingredients,
        totalCount=len(contexts),
        contexts=contexts,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8100, reload=True)
