import cv2
import numpy as np
from cv2 import aruco
import os

#Team ARise
#Team24 Aymaan K. , Anuj C. , Ethan M. , Neha S.
def findArucoMarkers(img, markerSize=6, totalMarkers=250, draw=True):
    """
    Detect ArUco markers in the image
    Updated to work with newer OpenCV-ArUco API
    """
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create ArUco dictionary based on marker size and total markers
    dictionary_name = f'DICT_{markerSize}X{markerSize}_{totalMarkers}'

    # Use the newer API for ArUco detection
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    # Detect markers
    corners, ids, rejected = detector.detectMarkers(imgGray)

    if draw and corners:
        cv2.aruco.drawDetectedMarkers(img, corners, ids)

    return [corners, ids]


def augmentAruco(bbox, id, img, imgAug, drawId=True):
    """
    Augment the ArUco marker with an image using homography
    """
    tl = bbox[0][0][0], bbox[0][0][1]
    tr = bbox[0][1][0], bbox[0][1][1]
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]

    h, w, c = imgAug.shape

    pts1 = np.array([tl, tr, br, bl])
    pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    # Finding homography matrix
    matrix, _ = cv2.findHomography(pts2, pts1)

    # Warping the augmentation image
    imgOut = cv2.warpPerspective(imgAug, matrix, (img.shape[1], img.shape[0]))

    # Create a mask for the region
    mask = np.zeros(img.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts1.astype(int), (255, 255, 255))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Apply mask to remove the original content
    img_masked = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))

    # Combine the original image and the augmented part
    imgOut = cv2.bitwise_and(imgOut, imgOut, mask=mask)
    imgOut = cv2.add(img_masked, imgOut)

    return imgOut


def loadOBJFile(file_path):
    """
    Load a 3D model from an OBJ file
    Returns vertices, faces, and texture coordinates
    """
    vertices = []
    faces = []
    texture_coords = []
    face_textures = []

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: 3D model file '{file_path}' not found.")

    try:
        # Try different encodings to handle special characters
        encodings = ['utf-8', 'latin-1', 'ISO-8859-1']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    for line in file:
                        if line.startswith('v '):  # Vertex
                            parts = line.strip().split()
                            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                        elif line.startswith('vt '):  # Texture coordinate
                            parts = line.strip().split()
                            texture_coords.append([float(parts[1]), float(parts[2])])
                        elif line.startswith('f '):  # Face
                            parts = line.strip().split()
                            face = []
                            face_texture = []
                            for part in parts[1:]:
                                indices = part.split('/')
                                if len(indices) >= 1:
                                    face.append(int(indices[0]) - 1)
                                if len(indices) >= 2 and indices[1]:
                                    face_texture.append(int(indices[1]) - 1)
                            faces.append(face)
                            if len(face_texture) > 0:
                                face_textures.append(face_texture)
                # If we get here without error, the file was read successfully
                break
            except UnicodeDecodeError:
                # If this encoding didn't work, try the next one
                continue

        if not vertices:
            raise ValueError("No valid data found in OBJ file with any of the attempted encodings")

    except Exception as e:
        raise Exception(f"Error reading OBJ file: {str(e)}")

    return np.array(vertices, dtype=np.float32), faces, np.array(texture_coords, dtype=np.float32), face_textures


def project3DModel(img, corners, camera_matrix, dist_coeffs, model_vertices, model_faces, texture_coords=None,
                   face_textures=None, texture_img=None):
    """
    Project a 3D model onto the ArUco marker with texture mapping
    Fixed to handle triangular faces properly
    """
    # Estimate the pose of the marker
    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, 0.05, camera_matrix, dist_coeffs)

    # Draw coordinate axes
    cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

    # Create a temporary image for rendering the model
    temp_img = img.copy()
    mask = np.zeros_like(img, dtype=np.uint8)

    # Check if texture is available
    has_texture = texture_img is not None and texture_coords is not None and face_textures and len(face_textures) > 0

    # Project 3D points to image plane
    for i, face in enumerate(model_faces):
        points_3d = model_vertices[face]

        # Project the 3D points to the image
        points_2d, _ = cv2.projectPoints(
            points_3d, rvec, tvec, camera_matrix, dist_coeffs)

        # Convert to integers
        points_2d = points_2d.reshape(-1, 2).astype(np.int32)

        if has_texture and i < len(face_textures):
            # Get texture coordinates for this face
            face_tex = face_textures[i]
            if len(face_tex) >= 3:  # Need at least 3 points for a face
                # Get texture coordinates for this face
                tex_points = texture_coords[face_tex]

                # Normalize texture coordinates
                tex_points = tex_points * np.array([texture_img.shape[1], texture_img.shape[0]])

                # Create source and destination points for perspective transform
                src_points = tex_points.astype(np.float32)
                dst_points = points_2d.astype(np.float32)

                # Check if we have enough points for homography (need at least 4)
                if len(src_points) == len(dst_points):
                    if len(src_points) >= 4:
                        # Use findHomography for 4+ points
                        try:
                            h, mask_h = cv2.findHomography(src_points, dst_points)

                            if h is not None:
                                # Create a warped texture
                                warped_texture = cv2.warpPerspective(
                                    texture_img, h, (img.shape[1], img.shape[0]))

                                # Create a mask for this face
                                face_mask = np.zeros_like(img, dtype=np.uint8)
                                cv2.fillConvexPoly(face_mask, points_2d, (255, 255, 255))

                                # Apply the mask to the warped texture
                                warped_face = cv2.bitwise_and(warped_texture, face_mask)

                                # Add to the mask for overall model
                                mask = cv2.bitwise_or(mask, face_mask)

                                # Add the textured face to the temporary image
                                temp_img = cv2.add(temp_img, warped_face)
                        except cv2.error:
                            # Fallback to solid color if homography fails
                            cv2.fillConvexPoly(mask, points_2d, (255, 255, 255))
                            cv2.fillConvexPoly(temp_img, points_2d, (0, 255, 0))  # Green color

                    elif len(src_points) == 3:
                        # For triangular faces (3 points), use Affine transform instead
                        #Mostly seen in low poly models
                        try:
                            # For triangles, use getAffineTransform which only needs 3 points
                            affine_matrix = cv2.getAffineTransform(
                                src_points[:3], dst_points[:3])

                            # Apply affine transform to the texture
                            warped_texture = cv2.warpAffine(
                                texture_img, affine_matrix, (img.shape[1], img.shape[0]))

                            # Create a mask for this face
                            face_mask = np.zeros_like(img, dtype=np.uint8)
                            cv2.fillConvexPoly(face_mask, points_2d, (255, 255, 255))

                            # Apply the mask to the warped texture
                            warped_face = cv2.bitwise_and(warped_texture, face_mask)

                            # Add to the mask for overall model
                            mask = cv2.bitwise_or(mask, face_mask)

                            # Add the textured face to the temporary image
                            temp_img = cv2.add(temp_img, warped_face)
                        except cv2.error:
                            # Fallback to solid color if affine transform fails
                            cv2.fillConvexPoly(mask, points_2d, (255, 255, 255))
                            cv2.fillConvexPoly(temp_img, points_2d, (120, 80, 200))  # Purple fallback
                    else:
                        # Not enough points, use solid color
                        cv2.fillConvexPoly(mask, points_2d, (255, 255, 255))
                        cv2.fillConvexPoly(temp_img, points_2d, (255, 128, 0))  # Orange fallback
                else:
                    # Mismatch in point counts, use solid color
                    cv2.fillConvexPoly(mask, points_2d, (255, 255, 255))
                    cv2.fillConvexPoly(temp_img, points_2d, (0, 255, 0))  # Green color
            else:
                # Not enough texture coordinates, use solid color
                cv2.fillConvexPoly(mask, points_2d, (255, 255, 255))
                cv2.fillConvexPoly(temp_img, points_2d, (0, 255, 0))  # Green color
        else:
            # No texture, use solid color
            cv2.fillConvexPoly(mask, points_2d, (255, 255, 255))
            cv2.fillConvexPoly(temp_img, points_2d, (0, 255, 0))  # Green color

    # Convert mask to grayscale
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Apply mask to original image (remove where model will be)
    img_masked = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask_gray))

    # Add the rendered model
    model_part = cv2.bitwise_and(temp_img, temp_img, mask=mask_gray)

    # Combine
    result = cv2.add(img_masked, model_part)

    return result


def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)

    # Load 2D image for standard augmentation
    img_path = r"C:\Users\USER\PycharmProjects\ARUCOMODULE\Markers\23.jpeg"
    if not os.path.exists(img_path):
        print(f"Warning: Image file '{img_path}' not found. Using a placeholder.")
        imgAug = np.zeros((100, 100, 3), dtype=np.uint8)
        imgAug[:, :, 1] = 255  # Green placeholder
    else:
        imgAug = cv2.imread(img_path)

    # Load 3D model - Changed to OBJ format
    model_path = r"C:\source\Carrot_3_LOD0\Carrot_3_LOD0.obj"  # Change to your model path
    texture_path = r"C:\source\Carrot_3_LOD0\untitled.png"  # Texture path
    use_3d_model = False
    texture_img = None

    # Load texture image if exists
    if os.path.exists(texture_path):
        texture_img = cv2.imread(texture_path)
        print(f"Loaded texture image of size {texture_img.shape if texture_img is not None else 'None'}")
    else:
        print(f"Warning: Texture file '{texture_path}' not found.")

    try:
        vertices, faces, texture_coords, face_textures = loadOBJFile(model_path)
        use_3d_model = True
        print(f"Loaded 3D model with {len(vertices)} vertices and {len(faces)} faces")
        print(f"Model has {len(texture_coords)} texture coordinates and {len(face_textures)} face texture mappings")
    except Exception as e:
        print(f"Could not load 3D model: {e}. Will use 2D augmentation instead.")
        texture_coords = None
        face_textures = None

    # Camera calibration parameters (replace with real calibration values)
    camera_matrix = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1]
    ], dtype=np.float32)

    dist_coeffs = np.zeros((5, 1), dtype=np.float32)

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            break

        # Find ArUco markers
        corners, ids = findArucoMarkers(img)

        # If markers are found
        if ids is not None:
            for i, (bbox, id) in enumerate(zip(corners, ids.flatten())):
                if use_3d_model:
                    img = project3DModel(img, np.array([bbox]), camera_matrix,
                                         dist_coeffs, vertices, faces,
                                         texture_coords, face_textures, texture_img)
                else:
                    img = augmentAruco(bbox, id, img, imgAug)

        # Display the result
        cv2.imshow('Augmented Reality', img)

        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()