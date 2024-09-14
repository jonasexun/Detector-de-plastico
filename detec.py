from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/color_detection', methods=['POST'])
def detect_plastic_by_color():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    image_file = request.files['image']
    
    if image_file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        # Leer la imagen en un array de bytes
        in_memory_image = np.frombuffer(image_file.read(), np.uint8)
        # Decodificar la imagen
        image = cv2.imdecode(in_memory_image, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Invalid image format"}), 400

        # Convertir la imagen a formato HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Definir el rango de color a detectar, excluyendo el azul
        lower_bound = np.array([0, 100, 100])  # Inferior del rango para colores no azules
        upper_bound = np.array([80, 255, 255]) # Superior del rango, evitando el azul

        # Crear una máscara para detectar el color en el rango definido
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

        # Contar el número de píxeles que coinciden
        color_count = cv2.countNonZero(mask)

        # Definir un umbral para considerar la presencia de plástico
        threshold = 1000  # Ajusta este valor según el contexto

        if color_count > threshold:
            detection_message = "Plastic detected"
        else:
            detection_message = "No plastic detected"

        return jsonify({"message": "Image received", "color_count": int(color_count), "detection": detection_message}), 200

    except Exception as e:
        # En caso de error, devolver un mensaje de error
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
