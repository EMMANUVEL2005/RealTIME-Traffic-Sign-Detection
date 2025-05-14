import cv2
import sys

def main():
    # Define cascade files and labels for multiple traffic signs
    cascades = {
        "stop_sign_classifier.xml": "Stop Sign"
        
    }

    # Load cascades
    loaded_cascades = {}
    for file, label in cascades.items():
        cascade = cv2.CascadeClassifier(file)
        if cascade.empty():
            print(f"Error loading cascade file '{file}'. Please ensure the file exists in the script directory.")
            sys.exit(1)
        loaded_cascades[file] = (cascade, label)

    # Initialize video capture from webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)

    print("Starting multi-sign traffic sign detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect and draw rectangles for each sign type
        for file, (cascade, label) in loaded_cascades.items():
            signs = cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            for (x, y, w, h) in signs:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Multi-Sign Traffic Sign Detection", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
