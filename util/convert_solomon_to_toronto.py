import json

def parse_and_convert_to_toronto(input_txt_path, output_json_path):
    customers_started = False
    points = []

    # Toronto reference point (downtown)
    center_lat = 43.651070
    center_lon = -79.347015
    x_scale = 0.02   # ~1 unit = ~2km lon
    y_scale = 0.02   # ~1 unit = ~2km lat

    with open(input_txt_path, "r") as file:
        for line in file:
            stripped = line.strip()
            if not customers_started:
                if stripped.startswith("CUST NO."):
                    customers_started = True
                    next(file)  # Skip separator
                    continue
            else:
                tokens = stripped.split()
                if len(tokens) < 7:
                    continue

                cust_id = int(tokens[0])
                if cust_id == 0:
                    continue  # Skip depot

                x = float(tokens[1])
                y = float(tokens[2])
                ready = int(tokens[4])
                due = int(tokens[5])

                lat = center_lat + (y - 50) * y_scale
                lon = center_lon + (x - 50) * x_scale

                points.append({
                    "id": cust_id,
                    "lat": round(lat, 6),
                    "lon": round(lon, 6),
                    "label": f"Customer {cust_id}",
                    "time_window": [ready, due]
                })

    with open(output_json_path, "w") as out_file:
        json.dump(points, out_file, indent=2)
    print(f"✅ Converted {len(points)} customers → {output_json_path}")


# 🧪 Example usage
if __name__ == "__main__":
    parse_and_convert_to_toronto("util/RC101.txt", "util/RC101.json")
