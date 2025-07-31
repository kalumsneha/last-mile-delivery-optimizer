import json
from types import SimpleNamespace


def convert_solomon_to_json(input_path, output_path):
    with open(input_path, 'r') as f:
        lines = f.readlines()

    customers = lines[9:]
    data = []

    for line in customers:
        tokens = line.strip().split()
        if len(tokens) != 7:
            continue
        cust_id = int(tokens[0])
        x = float(tokens[1])
        y = float(tokens[2])
        ready_time = int(tokens[4])
        due_date = int(tokens[5])
        time_window = [ready_time // 60, due_date // 60]
        data.append({
            "id": cust_id,
            "lat": y / 100.0,
            "lon": x / 100.0,
            "time_window": time_window if cust_id != 0 else None
        })

    base = data[0]
    points = data[1:]

    output = {
        "base": base,
        "points": points
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


# def load_solomon(input_path):
#     with open(input_path, 'r') as f:
#         lines = f.readlines()
#
#     customers = lines[9:]
#     points = []
#
#     for line in customers:
#         tokens = line.strip().split()
#         if len(tokens) != 7:
#             continue
#         cust_id = str(tokens[0])
#         x = float(tokens[1])
#         y = float(tokens[2])
#         ready_time = int(tokens[4])
#         due_date = int(tokens[5])
#         time_window = (ready_time // 60, due_date // 60) if cust_id != "0" else None
#
#         point = SimpleNamespace(
#             id=cust_id,
#             lat=y / 100.0,
#             lon=x / 100.0,
#             time_window=time_window
#         )
#         points.append(point)
#
#     base_point = points[0]
#     delivery_points = points[1:]
#     return base_point, delivery_points


def load_solomon(filepath):
    with open(filepath, 'r') as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("Expected JSON to be a list of customer points.")

    # Define a fixed depot location manually (you can update lat/lon)
    depot_dict = raw.pop(0)
    base = SimpleNamespace(
        id="DEPOT",
        lat=depot_dict["lat"],
        lon=depot_dict["lon"],
        demand=0
    )

    # Convert each customer dictionary into a SimpleNamespace
    points = [SimpleNamespace(**p) for p in raw]

    return base, points


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python solomon_loader.py <input_file.txt> <output_file.json>")
    else:
        convert_solomon_to_json(sys.argv[1], sys.argv[2])
