from flask import Flask, request, jsonify
import base64
import io
import onnx_inference
import cv2


app = Flask(__name__)
@app.route('/detect_text', methods=['POST'])
def detect_text():
    try:
        image_base64 = request.json.get('image_base64')


        # Convert image data to OpenCV format
        img = onnx_inference.base64_to_cv2(image_base64)

        # Perform inference using onnx_inference (assuming this function exists)
        result = onnx_inference.onnx_inference(img)

        # Draw boxes on the image (assuming draw_boxes and image_to_base64 functions exist)
        image = onnx_inference.draw_boxes(img, result)
        cv2.imwrite("./out.jpg", image)
        base64_result = onnx_inference.image_to_base64("./out.jpg")

    
            # Return the extracted text as a JSON response
        return jsonify({"result": result, "annotated_base64": base64_result})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host="172.20.10.3",port="3000",debug=True)
