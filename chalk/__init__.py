import attrs

@attrs.define
class SegmentationClass:
    name:str
    index:int
    render_color:tuple[int,int,int]

def hex_string_to_int_tuple(hex_string: str) -> tuple[int, int, int]:
    """Convert a hex color string to an RGB tuple."""
    hex_string = hex_string.lstrip('#')
    return tuple(int(hex_string[i:i+2], 16) for i in (0, 2, 4))

segmentation_classes = [
    SegmentationClass(name, i, hex_string_to_int_tuple(color)) for i, (name , color) in
    enumerate([
        ("background", "#000000"),
        ("chalk", "#FFFFFF"),
        ("sign_stop", "#FF0000"),
        ("sign_go", "#00FF00"),
        ("sign_turn_left", "#FCBA03"),
        ("sign_turn_right", "#C07835"),
        ("sign_u_turn", "#E3E63E"),
        ("sign_speed", "#62CF83"),
        ("sign_dump", "#3765AA"),
        ("edge", "#D965F0"),
    ])
]
assert len(set(l.name         for l in segmentation_classes)) == len(segmentation_classes), "Error - class names must be unique"
assert len(set(l.index        for l in segmentation_classes)) == len(segmentation_classes), "Error - class indices must be unique"
assert len(set(l.render_color for l in segmentation_classes)) == len(segmentation_classes), "Error - class colors must be unique"