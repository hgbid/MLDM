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
