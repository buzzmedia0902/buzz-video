// Detect faces in every JPEG of a folder with Apple Vision; print JSON lines {"f": name, "faces": [[x,y,w,h,conf,yaw], ...]}
// Coordinates are normalised with origin top-left.
import Foundation
import Vision
import AppKit

let dir = CommandLine.arguments[1]
let files = try FileManager.default.contentsOfDirectory(atPath: dir).filter { $0.hasSuffix(".jpg") }.sorted()
for name in files {
    let url = URL(fileURLWithPath: dir).appendingPathComponent(name)
    guard let img = NSImage(contentsOf: url), let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else { continue }
    let req = VNDetectFaceRectanglesRequest()
    let handler = VNImageRequestHandler(cgImage: cg, options: [:])
    try? handler.perform([req])
    var faces: [[Double]] = []
    for o in (req.results ?? []) {
        let b = o.boundingBox
        let yaw = o.yaw?.doubleValue ?? 0
        faces.append([Double(b.minX), Double(1 - b.maxY), Double(b.width), Double(b.height), Double(o.confidence), yaw])
    }
    let data = try JSONSerialization.data(withJSONObject: ["f": name, "faces": faces])
    print(String(data: data, encoding: .utf8)!)
}
