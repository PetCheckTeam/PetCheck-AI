import unittest

from ingredient_extractor import extract_ingredients


class RagPipelineTests(unittest.TestCase):
    def test_extract_ingredients_returns_ingredient_list(self):
        ocr_text = """
        제품명
        사용한 원료의 명칭
        닭고기, 쌀, 옥수수
        주의 사항
        개봉 후 냉장 보관
        """

        result = extract_ingredients(ocr_text)

        self.assertEqual(result, ["닭고기", "쌀", "옥수수"])

    def test_extract_ingredients_returns_empty_list_without_heading(self):
        result = extract_ingredients("제품명만 있고 원료 영역은 없습니다.")

        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
