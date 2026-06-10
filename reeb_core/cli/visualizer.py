"""VTK visualizer for the reeb_core pipeline.

Renders two side-by-side viewports:
  Left  – raw tractogram streamlines coloured by local orientation.
  Right – Reeb graph with:
            • Nodes coloured by topological class (degree-based).
            • Isolated nodes rendered semi-transparent.
            • Edges coloured by bundle-weight via a cool-to-warm LUT.
            • Scalar bar legend for edge weights.
            • Node-type legend overlay.
            • Navball orientation marker widget in both viewports.
            • Gradient background and viewport labels.
"""

import argparse
import sys
import vtk
import numpy as np
import networkx as nx

from reeb_core.utils import load_streamlines
from reeb_core.algorithms import construct_robust_reeb

# ---------------------------------------------------------------------------
# Colour palette (R, G, B in [0, 1])
# ---------------------------------------------------------------------------
_COLOUR_ISOLATED = (0.55, 0.55, 0.55)   # grey ghost
_COLOUR_TERMINAL = (1.00, 0.55, 0.26)   # warm orange  #FF8C42
_COLOUR_PASSING = (0.31, 0.76, 0.97)   # sky blue     #4FC3F7
_COLOUR_JUNCTION = (0.88, 0.25, 0.98)   # vivid magenta #E040FB

_OPACITY_ISOLATED = 0.15
_OPACITY_NORMAL = 1.00

# Node sphere radii per class
_RADIUS = {
    "isolated": 0.8,
    "terminal": 1.0,
    "passing":  1.2,
    "junction": 1.5,
}


# ===========================================================================
# Step 1 – Node classification
# ===========================================================================

def _classify_nodes(R: nx.Graph) -> dict:
    """Return a dict mapping each node id to its topological class string.

    Classes are inferred from node degree in the contracted Reeb graph:
        degree 0  →  "isolated"
        degree 1  →  "terminal"
        degree 2  →  "passing"
        degree ≥3 →  "junction"
    """
    classes = {}
    for n in R.nodes():
        d = R.degree(n)
        if d == 0:
            classes[n] = "isolated"
        elif d == 1:
            classes[n] = "terminal"
        elif d == 2:
            classes[n] = "passing"
        else:
            classes[n] = "junction"
    return classes


# ===========================================================================
# Step 2 – Per-class node polydata builders
# ===========================================================================

def _build_node_polydata_by_class(
    R: nx.Graph, node_loc: dict, node_classes: dict
) -> dict:
    """Build one vtkPolyData per node class, containing only that class's points.

    Returns
    -------
    dict[str → (vtkPolyData, {node_id → point_idx})]
    """
    class_names = ["isolated", "terminal", "passing", "junction"]
    polydatas = {c: (vtk.vtkPolyData(), {}) for c in class_names}

    # Accumulate points
    raw_pts = {c: vtk.vtkPoints() for c in class_names}

    for nid in R.nodes():
        cls = node_classes.get(nid, "passing")
        if nid not in node_loc:
            continue
        pos = node_loc[nid]
        pts = raw_pts[cls]
        idx = pts.GetNumberOfPoints()
        pts.InsertNextPoint(pos[0], pos[1], pos[2])
        polydatas[cls][1][nid] = idx

    for cls in class_names:
        pd, mapping = polydatas[cls]
        pd.SetPoints(raw_pts[cls])

    return polydatas


def _make_sphere_glyph(polydata: vtk.vtkPolyData, radius: float) -> vtk.vtkGlyph3D:
    """Return a vtkGlyph3D that stamps sphere glyphs at every point in polydata."""
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(radius)
    sphere.SetThetaResolution(16)
    sphere.SetPhiResolution(16)

    glyph = vtk.vtkGlyph3D()
    glyph.SetSourceConnection(sphere.GetOutputPort())
    glyph.SetInputData(polydata)
    glyph.ScalingOff()
    glyph.Update()
    return glyph


# ===========================================================================
# Step 3 – Edge polydata with cell-to-point scalar promotion
# ===========================================================================

def _build_edge_polydata(
    R: nx.Graph, node_loc: dict
) -> tuple:
    """Build a vtkPolyData for Reeb graph edges with weight scalars.

    Weights are stored as *cell* scalars.  A vtkCellDataToPointData filter
    is appended so that vtkTubeFilter can interpolate colours along tubes.

    Returns
    -------
    (vtkCellDataToPointData, vtkAlgorithmOutput, float, float)
        The filter object **must** be kept alive by the caller for the entire
        duration of the VTK interaction.  If it is garbage-collected, the
        pipeline's output port becomes a dangling pointer and VTK will segfault.
    """
    polydata = vtk.vtkPolyData()
    points = vtk.vtkPoints()

    node_id_to_idx = {}
    for idx, nid in enumerate(R.nodes()):
        if nid in node_loc:
            pos = node_loc[nid]
            points.InsertNextPoint(pos[0], pos[1], pos[2])
            node_id_to_idx[nid] = idx

    cells = vtk.vtkCellArray()
    weights = vtk.vtkFloatArray()
    weights.SetName("Weights")

    weight_values = []
    for u, v, data in R.edges(data=True):
        if u in node_id_to_idx and v in node_id_to_idx:
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, node_id_to_idx[u])
            line.GetPointIds().SetId(1, node_id_to_idx[v])
            cells.InsertNextCell(line)
            w = data.get("weight", 1.0)
            weights.InsertNextValue(w)
            weight_values.append(w)

    polydata.SetPoints(points)
    polydata.SetLines(cells)
    polydata.GetCellData().SetScalars(weights)

    w_min = float(min(weight_values)) if weight_values else 0.0
    w_max = float(max(weight_values)) if weight_values else 1.0
    if w_min == w_max:
        w_max = w_min + 1e-6

    # Promote cell scalars → point scalars so TubeFilter interpolates colour.
    # IMPORTANT: return the filter object itself so the caller can hold a
    # reference and prevent Python from garbage-collecting it mid-session.
    c2p = vtk.vtkCellDataToPointData()
    c2p.SetInputData(polydata)
    c2p.PassCellDataOn()
    c2p.Update()

    return c2p, c2p.GetOutputPort(), w_min, w_max


# ===========================================================================
# Step 4 – Edge LUT + scalar bar
# ===========================================================================

def _make_edge_lut(w_min: float, w_max: float) -> vtk.vtkLookupTable:
    """Create a cool-to-warm lookup table mapped to [w_min, w_max]."""
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    lut.SetRange(w_min, w_max)

    # Cool (blue) → white → warm (red)
    for i in range(256):
        t = i / 255.0
        if t < 0.5:
            s = t * 2.0
            r, g, b = s, s, 1.0
        else:
            s = (t - 0.5) * 2.0
            r, g, b = 1.0, 1.0 - s, 1.0 - s
        lut.SetTableValue(i, r, g, b, 1.0)
    lut.Build()
    return lut


def _make_scalar_bar(lut: vtk.vtkLookupTable) -> vtk.vtkScalarBarActor:
    """Create a scalar bar legend for edge weights, positioned bottom-right."""
    bar = vtk.vtkScalarBarActor()
    bar.SetLookupTable(lut)
    bar.SetTitle("Bundle\nWeight")
    bar.SetNumberOfLabels(5)
    bar.SetOrientationToVertical()

    # Normalised position within the right viewport (which occupies [0.5, 1.0]
    # of the window).  The scalar bar lives in its own renderer coordinate so
    # values are [0, 1] relative to that renderer.
    bar.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    bar.GetPositionCoordinate().SetValue(0.82, 0.06)
    bar.SetWidth(0.14)
    bar.SetHeight(0.40)

    # Style
    bar.GetTitleTextProperty().SetFontSize(11)
    bar.GetTitleTextProperty().SetColor(0.9, 0.9, 0.9)
    bar.GetTitleTextProperty().BoldOff()
    bar.GetLabelTextProperty().SetFontSize(9)
    bar.GetLabelTextProperty().SetColor(0.8, 0.8, 0.8)
    bar.GetLabelTextProperty().BoldOff()
    bar.DrawBackgroundOff()
    bar.DrawFrameOff()
    bar.DrawTickLabelsOn()

    return bar


# ===========================================================================
# Step 5 – Node type legend (text-based, robust across VTK versions)
# ===========================================================================

_LEGEND_ENTRIES = [
    ("●  Junction  (degree ≥ 3)", _COLOUR_JUNCTION),
    ("●  Passing   (degree = 2)", _COLOUR_PASSING),
    ("●  Terminal  (degree = 1)", _COLOUR_TERMINAL),
    ("●  Isolated  (degree = 0)", _COLOUR_ISOLATED),
]


def _make_node_legend(node_counts: dict = None) -> list:
    """Return a list of vtkTextActor objects forming a node-type legend.

    Parameters
    ----------
    node_counts : dict, optional
        Mapping from class name to integer count.  When provided, each
        legend row shows the count in parentheses.

    Callers should add each actor to the right renderer.
    The actors use normalised viewport coordinates so they stay fixed
    regardless of window size.
    """
    if node_counts is None:
        node_counts = {}
    actors = []
    # Title
    title = vtk.vtkTextActor()
    title.SetInput("Node Types  (by degree)")
    title.GetTextProperty().SetFontSize(11)
    title.GetTextProperty().BoldOn()
    title.GetTextProperty().SetColor(0.9, 0.9, 0.9)
    title.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    title.SetPosition(0.02, 0.94)
    actors.append(title)

    for i, (label, colour) in enumerate(_LEGEND_ENTRIES):
        cls_key = label.split()[1].lower()   # e.g. "junction", "passing" …
        count = node_counts.get(cls_key, 0)
        full_label = f"{label}  ({count})"
        actor = vtk.vtkTextActor()
        actor.SetInput(full_label)
        actor.GetTextProperty().SetFontSize(10)
        actor.GetTextProperty().SetColor(*colour)
        actor.GetTextProperty().BoldOff()
        actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        actor.SetPosition(0.02, 0.88 - i * 0.06)
        actors.append(actor)

    return actors


# ===========================================================================
# Step 7 – Navball orientation marker
# ===========================================================================

def _make_navball(interactor: vtk.vtkRenderWindowInteractor,
                  viewport: tuple) -> vtk.vtkOrientationMarkerWidget:
    """Create a single vtkOrientationMarkerWidget (navball).

    Only ONE instance per interactor is safe in VTK — having two causes a
    segfault via the shared internal observer.  Place this widget once and
    use ``_make_scene_axes`` to add a static axes actor to other renderers.

    Parameters
    ----------
    viewport : (x_min, y_min, x_max, y_max) in normalised *window* coords.
    """
    axes = vtk.vtkAxesActor()
    axes.SetShaftTypeToCylinder()
    axes.SetXAxisLabelText("X")
    axes.SetYAxisLabelText("Y")
    axes.SetZAxisLabelText("Z")
    for caption in (
        axes.GetXAxisCaptionActor2D(),
        axes.GetYAxisCaptionActor2D(),
        axes.GetZAxisCaptionActor2D(),
    ):
        caption.GetTextActor().GetTextProperty().SetFontSize(10)

    marker = vtk.vtkOrientationMarkerWidget()
    marker.SetOrientationMarker(axes)
    marker.SetInteractor(interactor)
    marker.SetViewport(*viewport)
    marker.SetEnabled(1)
    marker.InteractiveOff()
    return marker


def _make_scene_axes(scale: float = 5.0) -> vtk.vtkAxesActor:
    """Return a small axes actor placed at the world origin for a renderer.

    This is used as a static orientation guide in renderers that cannot host
    a second vtkOrientationMarkerWidget.
    """
    axes = vtk.vtkAxesActor()
    axes.SetShaftTypeToCylinder()
    axes.SetTotalLength(scale, scale, scale)
    axes.SetXAxisLabelText("X")
    axes.SetYAxisLabelText("Y")
    axes.SetZAxisLabelText("Z")
    for caption in (
        axes.GetXAxisCaptionActor2D(),
        axes.GetYAxisCaptionActor2D(),
        axes.GetZAxisCaptionActor2D(),
    ):
        caption.GetTextActor().GetTextProperty().SetFontSize(9)
        caption.GetTextActor().GetTextProperty().SetColor(0.8, 0.8, 0.8)
    return axes


# ===========================================================================
# Step 8 – Text labels & gradient background
# ===========================================================================

def _make_viewport_label(text: str) -> vtk.vtkTextActor:
    """Return a title text actor centred near the top of its renderer."""
    actor = vtk.vtkTextActor()
    actor.SetInput(text)
    actor.GetTextProperty().SetFontSize(14)
    actor.GetTextProperty().BoldOn()
    actor.GetTextProperty().SetColor(0.85, 0.85, 0.85)
    actor.GetTextProperty().SetJustificationToCentered()
    actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    actor.SetPosition(0.50, 0.96)
    return actor


def _setup_gradient_background(renderer: vtk.vtkRenderer,
                               top: tuple, bottom: tuple) -> None:
    """Apply a vertical gradient background to a renderer."""
    renderer.GradientBackgroundOn()
    renderer.SetBackground(*bottom)
    renderer.SetBackground2(*top)


# ===========================================================================
# Streamline helpers (unchanged logic, kept here for clarity)
# ===========================================================================

def _streamlines_to_vtk_polydata(streamlines):
    polydata = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    cells = vtk.vtkCellArray()
    current_pt_id = 0
    for sl in streamlines:
        n_pts = len(sl)
        if n_pts < 2:
            continue
        for pt in sl:
            points.InsertNextPoint(pt[0], pt[1], pt[2])
        polyline = vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(n_pts)
        for i in range(n_pts):
            polyline.GetPointIds().SetId(i, current_pt_id + i)
        cells.InsertNextCell(polyline)
        current_pt_id += n_pts
    polydata.SetPoints(points)
    polydata.SetLines(cells)
    return polydata


def _add_orientation_colors(polydata):
    points = polydata.GetPoints()
    n_points = points.GetNumberOfPoints()
    colors = vtk.vtkUnsignedCharArray()
    colors.SetName("Colors")
    colors.SetNumberOfComponents(3)
    colors.SetNumberOfTuples(n_points)
    lines = polydata.GetLines()
    lines.InitTraversal()
    id_list = vtk.vtkIdList()
    colors_data = np.zeros((n_points, 3), dtype=np.uint8)
    while lines.GetNextCell(id_list):
        n_ids = id_list.GetNumberOfIds()
        for i in range(n_ids - 1):
            pt_id1 = id_list.GetId(i)
            pt_id2 = id_list.GetId(i + 1)
            p1 = np.array(points.GetPoint(pt_id1))
            p2 = np.array(points.GetPoint(pt_id2))
            direction = p2 - p1
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction /= norm
            rgb = np.clip(np.abs(direction) * 255.0, 0, 255).astype(np.uint8)
            colors_data[pt_id1] = rgb
            if i == n_ids - 2:
                colors_data[pt_id2] = rgb
    for i in range(n_points):
        colors.SetTuple(i, (int(colors_data[i, 0]),
                            int(colors_data[i, 1]),
                            int(colors_data[i, 2])))
    polydata.GetPointData().SetScalars(colors)


# ===========================================================================
# Step 9 – Main run_visualizer
# ===========================================================================

def run_visualizer(streamlines, R, node_loc):
    """Build and launch the dual-viewport VTK window."""

    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(1400, 700)
    render_window.SetWindowName("reeb_core — Tractogram | Reeb Graph")

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    style = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(style)

    # -----------------------------------------------------------------------
    # LEFT renderer — streamlines
    # -----------------------------------------------------------------------
    renderer_left = vtk.vtkRenderer()
    renderer_left.SetViewport(0.0, 0.0, 0.5, 1.0)
    _setup_gradient_background(renderer_left,
                               top=(0.04, 0.04, 0.08),
                               bottom=(0.01, 0.01, 0.02))

    sl_poly = _streamlines_to_vtk_polydata(streamlines)
    _add_orientation_colors(sl_poly)

    sl_mapper = vtk.vtkPolyDataMapper()
    sl_mapper.SetInputData(sl_poly)

    sl_actor = vtk.vtkActor()
    sl_actor.SetMapper(sl_mapper)
    renderer_left.AddActor(sl_actor)

    renderer_left.AddActor2D(_make_viewport_label("Tractogram"))

    # -----------------------------------------------------------------------
    # RIGHT renderer — Reeb graph
    # -----------------------------------------------------------------------
    renderer_right = vtk.vtkRenderer()
    renderer_right.SetViewport(0.5, 0.0, 1.0, 1.0)
    _setup_gradient_background(renderer_right,
                               top=(0.04, 0.04, 0.08),
                               bottom=(0.01, 0.01, 0.02))
    # NOTE: depth peeling (order-independent transparency) is intentionally
    # omitted.  SetAlphaBitPlanes / SetMultiSamples mutate the framebuffer
    # configuration before the OpenGL context is created and crash on drivers
    # that do not support it (VTK probe: LastRenderingUsedDepthPeeling = 0).
    # Isolated nodes at opacity 0.15 still render correctly without it.

    if len(R.nodes()) > 0:
        node_classes = _classify_nodes(R)
        class_polydatas = _build_node_polydata_by_class(
            R, node_loc, node_classes)

        class_colours = {
            "isolated": _COLOUR_ISOLATED,
            "terminal": _COLOUR_TERMINAL,
            "passing":  _COLOUR_PASSING,
            "junction": _COLOUR_JUNCTION,
        }
        class_opacity = {
            "isolated": _OPACITY_ISOLATED,
            "terminal": _OPACITY_NORMAL,
            "passing":  _OPACITY_NORMAL,
            "junction": _OPACITY_NORMAL,
        }

        for cls in ["isolated", "terminal", "passing", "junction"]:
            pd, _ = class_polydatas[cls]
            if pd.GetNumberOfPoints() == 0:
                continue

            glyph = _make_sphere_glyph(pd, _RADIUS[cls])

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(glyph.GetOutputPort())
            mapper.ScalarVisibilityOff()

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*class_colours[cls])
            actor.GetProperty().SetOpacity(class_opacity[cls])
            # Slight specular for a polished look
            actor.GetProperty().SetAmbient(0.15)
            actor.GetProperty().SetDiffuse(0.70)
            actor.GetProperty().SetSpecular(0.30)
            actor.GetProperty().SetSpecularPower(30)
            renderer_right.AddActor(actor)

        # -------------------------------------------------------------------
        # Edges – coloured by weight
        # -------------------------------------------------------------------
        if len(R.edges) > 0:
            # Keep c2p_filter alive in this scope — if it were GC'd the port
            # would become a dangling pointer and VTK would segfault.
            c2p_filter, c2p_port, w_min, w_max = _build_edge_polydata(
                R, node_loc)

            lut = _make_edge_lut(w_min, w_max)

            tube = vtk.vtkTubeFilter()
            tube.SetInputConnection(c2p_port)
            tube.SetRadius(0.35)
            tube.SetNumberOfSides(8)
            tube.SetVaryRadiusToVaryRadiusOff()
            tube.Update()

            edges_mapper = vtk.vtkPolyDataMapper()
            edges_mapper.SetInputConnection(tube.GetOutputPort())
            edges_mapper.SetScalarModeToUsePointData()
            edges_mapper.SetScalarRange(w_min, w_max)
            edges_mapper.SetLookupTable(lut)
            edges_mapper.ScalarVisibilityOn()

            edges_actor = vtk.vtkActor()
            edges_actor.SetMapper(edges_mapper)
            edges_actor.GetProperty().SetAmbient(0.10)
            edges_actor.GetProperty().SetDiffuse(0.80)
            renderer_right.AddActor(edges_actor)

            # Scalar bar legend for edge weights
            scalar_bar = _make_scalar_bar(lut)
            renderer_right.AddActor2D(scalar_bar)

    # Node legend overlay — includes per-class counts
    if len(node_loc) > 0:
        from collections import Counter
        counts = Counter(node_classes.values())
    else:
        counts = {}
    for legend_actor in _make_node_legend(node_counts=dict(counts)):
        renderer_right.AddActor2D(legend_actor)

    renderer_right.AddActor2D(_make_viewport_label("Reeb Graph"))

    # -----------------------------------------------------------------------
    # Shared camera (synchronised rotation)
    # -----------------------------------------------------------------------
    shared_camera = renderer_left.GetActiveCamera()
    renderer_right.SetActiveCamera(shared_camera)

    render_window.AddRenderer(renderer_left)
    render_window.AddRenderer(renderer_right)

    renderer_left.ResetCamera()
    renderer_right.ResetCamera()

    render_window.Render()

    # -----------------------------------------------------------------------
    # Navball orientation marker
    #
    # vtkOrientationMarkerWidget is NOT safe to instantiate more than once per
    # interactor — a second instance triggers a segfault via its shared
    # internal observer.  We therefore create exactly ONE widget, placed in the
    # lower-left corner of the LEFT (tractogram) viewport.
    #
    # For the RIGHT (Reeb graph) viewport we add a plain vtkAxesActor directly
    # into the 3D scene — it rotates with the shared camera and is perfectly
    # readable without the widget machinery.
    # -----------------------------------------------------------------------
    navball = _make_navball(
        interactor,
        viewport=(0.00, 0.00, 0.10, 0.18)   # lower-left of left panel
    )

    # Static axes actor in the Reeb graph scene (right renderer)
    # Estimate a reasonable scale from the bounding box of node_loc
    if len(node_loc) > 0:
        coords = np.array(list(node_loc.values()), dtype=float)
        extent = np.max(coords, axis=0) - np.min(coords, axis=0)
        axes_scale = float(np.max(extent)) * \
            0.08 if np.max(extent) > 0 else 5.0
    else:
        axes_scale = 5.0
    scene_axes = _make_scene_axes(scale=axes_scale)
    renderer_right.AddActor(scene_axes)

    render_window.Render()
    interactor.Initialize()
    interactor.Start()


# ===========================================================================
# CLI entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified Reeb Graph Builder and VTK Visualizer CLI",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "in_tractogram", help="Path to input streamlines file (.trk or .tck)")
    parser.add_argument("--epsilon", "-e", type=float, default=1.5,
                        help="Distance threshold between streamlines to define bundles (default: 1.5)")
    parser.add_argument("--alpha", "-a", type=float, default=2.0,
                        help="Persistence spatial length threshold for graph contraction (default: 2.0)")
    parser.add_argument("--delta", "-d", type=int, default=2,
                        help="Minimum streamline count threshold to prune noisy bundles (default: 2)")
    parser.add_argument("--clustering_threshold", "-c", type=float, default=2.5,
                        help="MDF distance threshold in mm for Tractosearch clustering (default: 2.5)")
    parser.add_argument("--resample", "-r", type=int, default=40,
                        help="Number of points to resample streamlines (default: 40)")

    args = parser.parse_args()

    print(f"Loading streamlines from {args.in_tractogram}...")
    try:
        streamlines, header = load_streamlines(args.in_tractogram)
    except Exception as err:
        print(f"Error loading streamlines file: {err}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(streamlines)} streamlines.")
    print("Constructing Reeb graph using Tractosearch clustering...")
    try:
        R, node_loc = construct_robust_reeb(
            streamlines,
            eps=args.epsilon,
            alpha=args.alpha,
            delta=args.delta,
            clustering_threshold=args.clustering_threshold,
            resample_nb=args.resample
        )
    except Exception as err:
        print(f"Error constructing Reeb graph: {err}", file=sys.stderr)
        sys.exit(1)

    print(
        f"Reeb graph constructed: {len(R.nodes)} nodes, {len(R.edges)} edges.")

    # Print a brief node-type summary
    if len(R.nodes) > 0:
        classes = _classify_nodes(R)
        from collections import Counter
        counts = Counter(classes.values())
        print("Node types:")
        for cls in ["junction", "passing", "terminal", "isolated"]:
            print(f"  {cls:10s}: {counts.get(cls, 0)}")

    print("Launching VTK visualizer...")
    run_visualizer(streamlines, R, node_loc)


if __name__ == "__main__":
    main()
