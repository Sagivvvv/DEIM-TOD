'''
TinyPerson 测试集切块共享工具(离线切块脚本 + 评测器同源)。

离线切块脚本(tools/dataset/tile_tinyperson_test.py)与评测器(TinyPersonMergeEvaluator)
**共用**这里的 `build_tile_index`,以保证同一张整图在两侧生成的切块顺序、切块 id 与 corner
偏移完全一致(评测器据此把切块预测映射回原图并 merge)。两侧必须都吃**整图 GT** +
相同 tile_w/tile_h/overlap,映射才一致。

切块规则(动态均匀重叠,等价于训练集切块脚本的 overlap_and_starts):
- 切块尺寸 tile_w × tile_h(默认 640 × 576,宽 × 高)。
- overlap 参数语义为**最小保底重叠**(默认 50):对每个方向独立地按
    count = size // tile + 1
    ovl   = (tile*count - size) // (count - 1)
  计算,若 ovl < overlap(保底)则 count += 1 重算(多切一块换取更充裕重叠,避免小目标被
  边界切碎);stride = tile - ovl,起点 starts[i] = min(i*stride, size - tile)(末块 clamp 到
  边缘,消除因 ovl 向下取整可能留下的 1px 未覆盖条)。→ 每块重叠≈相等且 ≥ 保底值。
- 若原图某方向长度 <= 切块尺寸:不补黑边,单块取该方向实际范围 [0, length],偏移记 0。
'''

__all__ = ['generate_tile_corners', 'build_tile_index']


def _axis_starts(length, tile_len, min_overlap):
    """单方向切块:动态均匀重叠 + min_overlap 保底,返回 (overlap, starts)。

    等价于训练集切块脚本的 overlap_and_starts:
      count = length//tile + 1; overlap = (tile*count - length)//(count-1);
      若 overlap < min_overlap 则 count += 1 重算;stride = tile - overlap;
      starts[i] = min(i*stride, length - tile)(末块 clamp 到边缘)。
    length <= tile 时不补边,返回 (0, [0])(实际尺寸由调用方 min(tile, length) 得出)。
    """
    length = int(length)
    tile_len = int(tile_len)
    min_overlap = int(min_overlap)

    # 原图该方向不大于切块尺寸:不补边,单块起点 0
    if length <= tile_len:
        return 0, [0]

    count = length // tile_len + 1
    overlap = (tile_len * count - length) // (count - 1)
    # 重叠不足保底值:多切一块换取更充裕的重叠,保证小目标不被边界切碎
    while overlap < min_overlap:
        count += 1
        overlap = (tile_len * count - length) // (count - 1)

    stride = tile_len - overlap
    if stride <= 0:
        raise ValueError(
            '无法计算有效切块步长:尺寸=%d, 切块=%d, 重叠=%d' % (length, tile_len, overlap))

    last_start = length - tile_len
    starts = [min(i * stride, last_start) for i in range(count)]
    return overlap, starts


def generate_tile_corners(width, height, tile_w=640, tile_h=576, overlap=50):
    """
    生成整图的所有切块 corner(原图像素坐标),行优先(row=y 外层, col=x 内层)。

    Args:
        width, height (int): 原图宽、高。
        tile_w, tile_h (int): 切块宽、高(默认 640 × 576)。
        overlap (int): 横/纵**最小保底重叠**像素(默认 50);实际重叠按图像尺寸动态计算(≥此值)。

    Return:
        List[Tuple[int, int, int, int]]: [(x1, y1, x2, y2), ...]
    """
    width, height = int(width), int(height)
    tile_w, tile_h = int(tile_w), int(tile_h)

    _ovl_w, xs = _axis_starts(width, tile_w, overlap)
    _ovl_h, ys = _axis_starts(height, tile_h, overlap)

    corners = []
    for y in ys:
        y2 = y + min(tile_h, height - y)      # 末块/小图 clamp 到实际边缘
        for x in xs:
            x2 = x + min(tile_w, width - x)
            corners.append((int(x), int(y), int(x2), int(y2)))
    return corners


def build_tile_index(coco, tile_w=640, tile_h=576, overlap=50):
    """
    由整图 COCO 构建确定性的切块索引。数据集与评测器共用,保证一致。

    Args:
        coco: 整图标注 COCO(faster_coco_eval 或原生 pycocotools 均可,
              需支持 getImgIds() 与 .imgs[id] 的 width/height/file_name)。
        tile_w, tile_h, overlap: 见 generate_tile_corners。

    Return:
        List[Tuple[int, int, str, Tuple[int, int, int, int]]]:
            [(tile_id, orig_img_id, file_name, (x1, y1, x2, y2)), ...],
            tile_id 从 1 连续编号(按 sorted(整图 id) 顺序 + 行优先切块顺序)。
    """
    tiles = []
    tile_id = 1
    for img_id in sorted(int(i) for i in coco.getImgIds()):
        info = coco.imgs[img_id]
        w = int(info['width'])
        h = int(info['height'])
        file_name = info['file_name']
        for corner in generate_tile_corners(w, h, tile_w, tile_h, overlap):
            tiles.append((tile_id, img_id, file_name, corner))
            tile_id += 1
    return tiles
