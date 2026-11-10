"""Small explicit arithmetic verifier, not natural-language understanding."""
import re
import unicodedata


def parse_integer_expression(text):
    text=unicodedata.normalize('NFKC',text).lower()
    numbers=re.findall(r'(?<!\d)\d{1,2}(?!\d)',text)
    if len(numbers)!=2 or 'không' in text or 'not ' in text:return None
    direct=re.search(r'(?<!\d)(\d{1,2})\s*(\+|\-|−|cộng|trừ|plus|minus)\s*(\d{1,2})(?!\d)',text)
    if direct:
        a,b=int(direct[1]),int(direct[3]);op='+' if direct[2] in ('+','cộng','plus') else '-'
        return a,op,b
    quantity=re.search(r'\bcó\s+(\d{1,2})\b[^\d]{1,60}?(thêm|bớt(?:\s+đi)?)\s+(\d{1,2})\b',text)
    if quantity:return int(quantity[1]),'+' if quantity[2]=='thêm' else '-',int(quantity[3])
    return None


def integer_answer(expression):
    a,op,b=expression
    return a+b if op=='+' else a-b
