import cv2
import numpy as np
import mediapipe as mp
import dlib


relative = lambda landmark, shape: (int(landmark.x * shape[1]), int(landmark.y * shape[0]))
relativeT = lambda landmark, shape: (int(landmark.x * shape[1]), int(landmark.y * shape[0]), 0)

def draw_face_mesh(frame, face_mesh_landmarks):
    for landmark in face_mesh_landmarks:
        for point in landmark.landmark:
            x, y = relative(point, frame.shape)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

def gaze(frame, points):
    """
    The gaze function gets an image and face landmarks from the mediapipe framework.
    The function draws the gaze direction into the frame.
    """

    # 2D image points.
    image_points = np.array([
        relative(points.landmark[4], frame.shape),  # Nose tip
        relative(points.landmark[152], frame.shape),  # Chin
        relative(points.landmark[263], frame.shape),  # Left eye left corner
        relative(points.landmark[33], frame.shape),  # Right eye right corner
        relative(points.landmark[287], frame.shape),  # Left Mouth corner
        relative(points.landmark[57], frame.shape)  # Right mouth corner
    ], dtype="double")

    # 2D image points with Z=0.
    image_points1 = np.array([
        relativeT(points.landmark[4], frame.shape),  # Nose tip
        relativeT(points.landmark[152], frame.shape),  # Chin
        relativeT(points.landmark[263], frame.shape),  # Left eye, left corner
        relativeT(points.landmark[33], frame.shape),  # Right eye, right corner
        relativeT(points.landmark[287], frame.shape),  # Left Mouth corner
        relativeT(points.landmark[57], frame.shape)  # Right mouth corner
    ], dtype="double")

    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0, -63.6, -12.5),  # Chin
        (-43.3, 32.7, -26),  # Left eye, left corner
        (43.3, 32.7, -26),  # Right eye, right corner
        (-28.9, -28.9, -24.1),  # Left Mouth corner
        (28.9, -28.9, -24.1)  # Right mouth corner
    ])

    # 3D model eye points - The center of the eye ball
    Eye_ball_center_right = np.array([[-29.05], [32.7], [-39.5]])
    Eye_ball_center_left = np.array([[29.05], [32.7], [-39.5]])

    # Camera matrix estimation.
    focal_length = frame.shape[1]
    center = (frame.shape[1] / 2, frame.shape[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # 2D pupil location.
    left_pupil = relative(points.landmark[468], frame.shape)

    # Transformation between image point to world point.
    _, transformation, _ = cv2.estimateAffine3D(image_points1, model_points)  # Image to world transformation

    if transformation is not None:  # If estimateAffine3D succeeded.
        # Project pupil image point into 3D world point.
        pupil_world_cord = transformation @ np.array([[left_pupil[0], left_pupil[1], 0, 1]]).T

        # 3D gaze point (10 is an arbitrary value denoting gaze distance).
        S = Eye_ball_center_left + (pupil_world_cord - Eye_ball_center_left) * 10

        # Project a 3D gaze direction onto the image plane.
        (eye_pupil2D, _) = cv2.projectPoints(np.array([[S[0], S[1], S[2]]]), rotation_vector,
                                             translation_vector, camera_matrix, dist_coeffs)

        # Project 3D head pose into the image plane.
        (head_pose, _) = cv2.projectPoints(pupil_world_cord.reshape(1, 3), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        # Correct gaze for head rotation.
        gaze = left_pupil + (eye_pupil2D[0][0] - left_pupil) - (head_pose[0][0] - left_pupil)

        # Draw gaze line into the screen.
        p1 = (int(left_pupil[0]), int(left_pupil[1]))
        p2 = (int(gaze[0]), int(gaze[1]))
        cv2.line(frame, p1, p2, (0, 0, 255), 2)

    # Draw face mesh on the frame
        if face_mesh_landmarks:
            draw_face_mesh(frame, face_mesh_landmarks)    


# Initialize the face mesh model from Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Camera stream:
cap = cv2.VideoCapture(0)  # choose camera index (try 1, 2, 3)

while cap.isOpened():
    success, image = cap.read()

    if not success:
        print("Ignoring empty camera frame.")
        continue

    # To improve performance, optionally mark the image as not writeable to pass by reference.
    image.flags.writeable = False
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process face mesh
    results = face_mesh.process(image_rgb)
    face_mesh_landmarks = results.multi_face_landmarks

    # Process gaze
    image.flags.writeable = True
    gaze(image, results.multi_face_landmarks[0] if face_mesh_landmarks else None)

    # Draw output
    if face_mesh_landmarks:
        draw_face_mesh(image, face_mesh_landmarks)

    cv2.imshow('output window', image)
    if cv2.waitKey(2) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
