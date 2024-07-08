import ast
import re

number = {
    "קרקע": 0,"ק":0,
    "ראשונה": 1, "אחת": 1,
    "שניה": 2, "שתים": 2, "שתיים": 2,
    "שלישית": 3, "שלוש": 3, "שלישי": 3,
    "רביעית": 4, "ארבע": 4, "רביעי": 4,
    "חמישית": 5, "חמש": 5, "חמישי": 5,
    "שישית": 6, "שש": 6, "שישי": 6,
    "שביעית": 7, "שבע": 7, "שביעי": 7,
    "שמינית": 8, "שמונה": 8, "שמיני": 8,
    "תשיעית": 9, "תשע": 9, "תשיעי": 9,
    "עשירית": 10, "עשר": 10, "עשירי": 10,
    "אחת עשרה": 11, "אחד עשר": 11,
    "שתים עשרה": 12, "שנים עשר": 12,
    "שלוש עשרה": 13, "שלושה עשר": 13,
    "ארבע עשרה": 14, "ארבעה עשר": 14,
    "חמש עשרה": 15, "חמישה עשר": 15,
    "שש עשרה": 16, "שישה עשר": 16,
    "שבע עשרה": 17, "שבעה עשר": 17,
    "שמונה עשרה": 18, "שמונה עשר": 18,
    "תשע עשרה": 19, "תשעה עשר": 19,
    "עשרים": 20,
    "עשרים ואחת": 21, "עשרים ואחת": 21,
    "עשרים ושתיים": 22, "עשרים ושניים": 22,
    "עשרים ושלוש": 23, "עשרים ושלושה": 23,
    "עשרים וארבע": 24, "עשרים וארבעה": 24,
    "עשרים וחמש": 25, "עשרים וחמישה": 25,
    "עשרים ושש": 26,
    "עשרים ושבע": 27, "עשרים ושבעה": 27,
    "עשרים ושמונה": 28,
    "עשרים ותשע": 29, "עשרים ותשעה": 29,
    "שלושים": 30}


def get_floor_number(s):
    match = re.search(r'\d+', s)
    if match:
        return int(match.group())
    first = re.split(r'\+|,', s)[0].replace(",", "").replace(".", "")
    if first in number:
        print(number[first])
        return number[first]
    return number.get(s, None)


def extract_numbers(s):
    if not s:
        return 0
    numeric_string = re.sub(r'\D', '', str(s))
    if numeric_string:
        return int(numeric_string)
    else:
        return 0


def extract_coordinates(coord_str):
    try:
        coord_dict = ast.literal_eval(coord_str)
        return coord_dict.get('latitude', 0), coord_dict.get('longitude', 0)
    except (ValueError, SyntaxError):
        return 0, 0


def get_data_from_search_text(search_text):
    result = {key: 0 for key in text_keywords}

    if search_text is None or not isinstance(search_text, str):
        return result

    parts = search_text.split('כולל')
    if len(parts) == 1:
        return result

    for part in parts[1:]:
        words = part.split()
        for word in words:
            for key, keyword in text_keywords.items():
                if keyword in word:
                    result[key] = 1

    return result
