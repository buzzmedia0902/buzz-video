// Text lines of every JPEG of a folder with Apple Vision text recognition (Japanese + English); one JSON line per image:
//   {"f": name, "lines": [{"text": "...", "box": [x, y, w, h], "conf": 0.9}]}
// box is normalised with origin top-left. Used by ref_probe.py to measure where a reference video puts its telops,
// how big they are and what colour they are.
import AppKit
import Foundation
import Vision

let dir = CommandLine.arguments[1]
let files = try FileManager.default.contentsOfDirectory(atPath: dir).filter { $0.hasSuffix(".jpg") }.sorted()
for name in files {
    let url = URL(fileURLWithPath: dir).appendingPathComponent(name)
    guard let img = NSImage(contentsOf: url), let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else { continue }
    let req = VNRecognizeTextRequest()
    req.recognitionLevel = .accurate
    req.recognitionLanguages = ["ja-JP", "en-US"]
    req.usesLanguageCorrection = false      // telops are stylised words: do not "correct" them
    req.minimumTextHeight = 0.02
    let handler = VNImageRequestHandler(cgImage: cg, options: [:])
    try? handler.perform([req])
    var lines: [[String: Any]] = []
    for o in (req.results ?? []) {
        guard let top = o.topCandidates(1).first else { continue }
        let b = o.boundingBox
        lines.append(["text": top.string, "conf": Double(top.confidence),
                      "box": [Double(b.minX), Double(1 - b.maxY), Double(b.width), Double(b.height)]])
    }
    let data = try JSONSerialization.data(withJSONObject: ["f": name, "lines": lines])
    print(String(data: data, encoding: .utf8)!)
}
