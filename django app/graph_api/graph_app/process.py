
from models import Order

def get_entries():
    # get last 10 order
    if len(Order.objects) > 10:
        last_ten = Order.objects[:-10]
    last_ten = Order.objects

    entries = []
    for entry in reversed(last_ten):
        entries.append(entry)

    return entries

s = get_entries()