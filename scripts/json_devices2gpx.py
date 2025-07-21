import json
from datetime import datetime
from xml.etree.ElementTree import Element, SubElement, tostring, indent
from pathlib import Path
import sys

def create_gpx_file(json_file, gpx_file=None):
    # Load the JSON data
    json_file = Path(json_file)
    with json_file.open("r", encoding="utf-8") as file:
        data = json.load(file)

    # Create the GPX root element
    gpx = Element(
        "gpx",
        attrib={
            "version": "1.0",
            "creator": "ao.korzh@gmail.com",
            "xmlns": "http://www.topografix.com/GPX/1/0",
        },
    )

    # Iterate through the JSON data and create waypoints
    i_point_name = 0
    i_elevation = 1
    i_comment = 2
    i_coord = 4
    i_time = 6
    waypoint_name_prev = None
    for device_name, waypoint_data in data.items():
        print(device_name, waypoint_data, sep=": ")
        waypoint_name = waypoint_data[i_point_name]
        if (not device_name) or device_name.startswith(" "):  #
            continue
        if waypoint_name_prev == waypoint_name:
            # cmt_el = wp_el.find("cmt")
            cmt_el.text = cmt_el.text + "," + device_name
            continue
        wp_el = SubElement(
            gpx,
            "wpt",
            attrib={
                "lat": str(waypoint_data[i_coord]),
                "lon": str(waypoint_data[i_coord + 1]),
            },
        )
        # time first
        try:
            time = waypoint_data[i_time]
            if time:
                SubElement(wp_el, "time").text = datetime.fromisoformat(time).strftime("%Y-%m-%dT%H:%M:%SZ")
        except IndexError:
            pass

        # next in alphabetical order:

        # Elevation (we write depth)
        ele = waypoint_data[i_elevation]
        if ele:
            SubElement(wp_el, "ele").text = str(ele)

        # waypoint name
        SubElement(wp_el, "name").text = waypoint_name

        # Comment
        cmt_el = SubElement(wp_el, "cmt")
        cmt_el.text = device_name

        # desc
        other = [str(e) for e in waypoint_data[i_comment:i_coord] if e]
        if other:
            SubElement(wp_el, "desc").text = ' '.join(other)


        waypoint_name_prev = waypoint_name

    indent(gpx)  # for pretty printing to file

    # Write the GPX file
    output_gpx = json_file.with_suffix(".gpx") if gpx_file is None else Path(gpx_file)
    with output_gpx.open("w", encoding="utf-8") as file:
        file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        file.write(tostring(gpx, encoding="utf-8").decode("utf-8"))

    print(f"saved to {output_gpx.name}")

if __name__ == "__main__":
    n_args = len(sys.argv) - 1
    print(n_args, 'arguments:', sys.argv[1:])
    if n_args < 1:
        print(
            f"usage:\n{(Path(__file__).name)} file_in.json [file_out.gpx]\n"
            "Where file_in.json should contain structure: {\n"
            '"device_name": ["point_name", depth, 0.5, "↟", Latitude°, Longitude°, "2024-11-15T12:57"],\n'
            "...} Where:\n"
            '"device_name": gpx "comment",\n'
            '"point_name": gpx "name",\n'
            '"depth": gpx "elevation",\n'
            "Latitude and Longitude: gpx coordinates.\n"
            'Other fields goes to "desc"\n'
        )
        exit(0)

    filename_in = sys.argv[1]
    if n_args > 1:
        file_out = sys.argv[2]
    else:
        file_out = None
    create_gpx_file(filename_in, file_out)
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} Ok>")