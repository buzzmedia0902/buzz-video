// Facial expression numbers with Apple Vision: print one JSON line per JPEG of a folder (names sorted)
//   {"f": name, "faces": [{"box": [x, y, w, h], "conf": c, "roll": r, "yaw": y, "pitch": p,
//                          "eye": open ratio (eye height / width, mean of both), "eye_l": ..., "eye_r": ...,
//                          "mouth_open": inner lips height / face height, "mouth_w": outer lips width / face width,
//                          "lift": mouth corners above the lip centre / face height (> 0 smiles),
//                          "brow": eyebrows above the eyes / face height}]}
// box is normalised with origin top-left (like faces.swift); every other number is a ratio. Used by thumb.py to pick
// the face of a thumbnail: eyes open, a real smile, an open mouth for a reaction.
import AppKit
import Foundation
import Vision

func extent(_ pts: [CGPoint]) -> (w: Double, h: Double) {
    guard let x0 = pts.map({ $0.x }).min(), let x1 = pts.map({ $0.x }).max(),
          let y0 = pts.map({ $0.y }).min(), let y1 = pts.map({ $0.y }).max() else { return (0, 0) }
    return (Double(x1 - x0), Double(y1 - y0))
}

func meanY(_ pts: [CGPoint]) -> Double {
    pts.isEmpty ? 0 : Double(pts.map { $0.y }.reduce(0, +)) / Double(pts.count)
}

let dir = CommandLine.arguments[1]
let files = try FileManager.default.contentsOfDirectory(atPath: dir).filter { $0.hasSuffix(".jpg") }.sorted()
for name in files {
    let url = URL(fileURLWithPath: dir).appendingPathComponent(name)
    guard let img = NSImage(contentsOf: url), let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else { continue }
    let req = VNDetectFaceLandmarksRequest()
    let handler = VNImageRequestHandler(cgImage: cg, options: [:])
    try? handler.perform([req])
    var faces: [[String: Any]] = []
    for o in (req.results ?? []) {
        let b = o.boundingBox
        let bw = Double(b.width) * Double(cg.width), bh = Double(b.height) * Double(cg.height)
        // landmark points are normalised to the face box: scale to pixels so the ratios keep their aspect
        func px(_ r: VNFaceLandmarkRegion2D?) -> [CGPoint] {
            (r?.normalizedPoints ?? []).map { CGPoint(x: Double($0.x) * bw, y: Double($0.y) * bh) }
        }
        let el = px(o.landmarks?.leftEye), er = px(o.landmarks?.rightEye)
        let outer = px(o.landmarks?.outerLips), inner = px(o.landmarks?.innerLips)
        let brows = px(o.landmarks?.leftEyebrow) + px(o.landmarks?.rightEyebrow)
        let (lw, lh) = extent(el), (rw, rh) = extent(er)
        let eyeL = lw > 0 ? lh / lw : 0, eyeR = rw > 0 ? rh / rw : 0
        var lift = 0.0
        if let left = outer.min(by: { $0.x < $1.x }), let right = outer.max(by: { $0.x < $1.x }) {
            lift = (Double(left.y + right.y) / 2 - meanY(outer)) / bh
        }
        let eyeY = meanY(el + er)
        faces.append([
            "box": [Double(b.minX), Double(1 - b.maxY), Double(b.width), Double(b.height)],
            "conf": Double(o.confidence),
            "roll": o.roll?.doubleValue ?? 0, "yaw": o.yaw?.doubleValue ?? 0, "pitch": o.pitch?.doubleValue ?? 0,
            "eye": (eyeL + eyeR) / 2, "eye_l": eyeL, "eye_r": eyeR,
            "mouth_open": extent(inner).h / bh,
            "mouth_w": extent(outer).w / bw,
            "lift": lift,
            "brow": brows.isEmpty || el.isEmpty ? 0 : (meanY(brows) - eyeY) / bh,
        ])
    }
    let data = try JSONSerialization.data(withJSONObject: ["f": name, "faces": faces])
    print(String(data: data, encoding: .utf8)!)
}
