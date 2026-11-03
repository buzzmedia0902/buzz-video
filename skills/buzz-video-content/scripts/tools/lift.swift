// Subject cut-out masks with Apple Vision (the same model as "Lift subject" in Photos).
//
//   swiftc -O tools/lift.swift -o lift && lift in.png out_prefix [in2.png out_prefix2 ...]
//   -> out_prefix_fg<N>.png   one soft mask per foreground instance (full resolution, 8-bit gray)
//      out_prefix_person.png  person matte (VNGeneratePersonSegmentationRequest, accurate; native size)
// Several in / prefix pairs share one process: the models load once (about 0.3 s per frame instead of 2.5 s).
import CoreImage
import Foundation
import Vision

func writeMask(_ buffer: CVPixelBuffer, to path: String, context: CIContext) throws {
    let image = CIImage(cvPixelBuffer: buffer)
    try context.writePNGRepresentation(of: image, to: URL(fileURLWithPath: path), format: .L8,
                                       colorSpace: CGColorSpaceCreateDeviceGray())
}

func lift(_ inputPath: String, _ prefix: String, context: CIContext) -> Bool {
    guard let input = CIImage(contentsOf: URL(fileURLWithPath: inputPath)) else {
        FileHandle.standardError.write("cannot read \(inputPath)\n".data(using: .utf8)!)
        return false
    }
    let handler = VNImageRequestHandler(ciImage: input, options: [:])
    let foreground = VNGenerateForegroundInstanceMaskRequest()
    let person = VNGeneratePersonSegmentationRequest()
    person.qualityLevel = .accurate
    person.outputPixelFormat = kCVPixelFormatType_OneComponent8
    do {
        try handler.perform([foreground, person])
        if let obs = foreground.results?.first {
            for instance in obs.allInstances {
                let mask = try obs.generateScaledMaskForImage(forInstances: IndexSet(integer: instance), from: handler)
                try writeMask(mask, to: "\(prefix)_fg\(instance).png", context: context)
            }
            print("\(prefix) foreground instances:", obs.allInstances.map { $0 })
        } else {
            print("\(prefix) foreground: no subject")
        }
        if let obs = person.results?.first {
            try writeMask(obs.pixelBuffer, to: "\(prefix)_person.png", context: context)
            print("\(prefix) person matte:", CVPixelBufferGetWidth(obs.pixelBuffer), "x",
                  CVPixelBufferGetHeight(obs.pixelBuffer))
        }
        return true
    } catch {
        FileHandle.standardError.write("vision failed on \(inputPath): \(error)\n".data(using: .utf8)!)
        return false
    }
}

let args = CommandLine.arguments
guard args.count >= 3, args.count % 2 == 1 else {
    FileHandle.standardError.write("usage: lift in.png out_prefix [in2.png out_prefix2 ...]\n".data(using: .utf8)!)
    exit(2)
}
let context = CIContext()
var failed = false
for i in stride(from: 1, to: args.count, by: 2) where !lift(args[i], args[i + 1], context: context) {
    failed = true
}
exit(failed ? 1 : 0)
