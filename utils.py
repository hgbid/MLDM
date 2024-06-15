import ast
import re

condition_dict = {
    'חדש מקבלן (לא גרו בנכס)': 1,
    'חדש (גרו בנכס)': 2,
    'משופץ': 3,
    'במצב שמור': 4,
    'דרוש שיפוץ': 5
}
ad_highlight = {
    'none': 1,
    'pink': 2,
    'yellow': 3
}
text_keywords = {
    "safe_room": 'ממ"ד',
    "elevator": 'מעלית',
    "parking": 'חניה',
    "bars": 'סורגים',
    "air_conditioning": 'מיזוג',
    "air_conditioner": 'מזגן',
    "accessible": 'לנכים',
    "furniture": 'ריהוט'
}


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