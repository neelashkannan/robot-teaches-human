"""Stub: cycles the teacher hand through open, fist, index-only so it can be seen. q quits.
The real webcam + marking loop replaces this at 13:15."""
import time

import cv2
import numpy as np

from hand.teacher import Teacher

POSES = [
    ("open", np.zeros(8)),
    ("fist", np.array([1.0, 0.0] * 4)),
    ("index only", np.array([1.0, 0, 0, 0, 1, 0, 1, 0])),
]


def main() -> None:
    teacher = Teacher()
    t0 = time.time()
    last = -1
    while True:
        k = int((time.time() - t0) / 2.0) % len(POSES)
        name, vec = POSES[k]
        if k != last:
            teacher.set_pose(vec)
            last = k
        teacher.step(20)
        frame = cv2.cvtColor(teacher.render(480, 480), cv2.COLOR_RGB2BGR)
        cv2.putText(frame, name, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.imshow("teacher", frame)
        if cv2.waitKey(16) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
