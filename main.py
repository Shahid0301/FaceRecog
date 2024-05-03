import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime, timedelta

path = 'Training_images'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    # Dictionary to store last attendance time for each person
    last_attendance = {}

    # Read existing attendance records from the CSV file and update last_attendance
    with open('Attendance.csv', 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 3:
                last_attendance[parts[0]] = (parts[1], parts[2])

    # Get the current date and time
    now = datetime.now()
    current_date = now.strftime('%Y-%m-%d')
    current_time = now.strftime('%H:%M:%S')

    # Check if the name is not in last_attendance or the last attendance was more than 5 minutes ago
    if name not in last_attendance or (datetime.strptime(current_time, '%H:%M:%S') - datetime.strptime(last_attendance[name][1], '%H:%M:%S') > timedelta(minutes=5)):
        # Write attendance record to the CSV file
        with open('Attendance.csv', 'a') as f:
            f.write(f'\n{name},{current_date},{current_time}')
        
        # Update last_attendance for the person
        last_attendance[name] = (current_date, current_time)


encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

attendance_marked = False  # Flag to keep track of whether attendance has been marked

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
            
            if not attendance_marked:  # Check if attendance has not been marked yet
                markAttendance(name)
                attendance_marked = True  # Set the flag to True to indicate attendance has been marked
            
    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit the loop if 'q' is pressed
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()