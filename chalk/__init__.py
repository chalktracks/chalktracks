import attrs

@attrs.define
class SegmentationClass:
    name:str
    index:int
    render_color:tuple[int,int,int]

segmentation_classes = [
    SegmentationClass(name, index, color) for name, index, color in
    [
        ("chalk", 1, (255,255,255)),
        ("sign_stop", 2, (255,0,0)),
        ("sign_turn", 3, (252, 186, 3)),
        ("edge", 4, (0,255,0)),
    ]
]
assert len(set(l.name         for l in segmentation_classes)) == len(segmentation_classes), "Error - class names must be unique"
assert len(set(l.index        for l in segmentation_classes)) == len(segmentation_classes), "Error - class indices must be unique"
assert len(set(l.render_color for l in segmentation_classes)) == len(segmentation_classes), "Error - class colors must be unique"