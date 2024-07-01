if [ ! -d "./data" ]; then
  tar -xvzf './landmarks_task.tgz'
  mv ./landmarks_task ./data
fi

if [ ! -d "./model_weights" ]; then
    mkdir -p ./model_weights
    wget -O ./model_weights/mmod_human_face_detector.dat.bz2 http://dlib.net/files/mmod_human_face_detector.dat.bz2 
    bzip2 -dk ./model_weights/mmod_human_face_detector.dat.bz2

    wget -O ./model_weights/shape_predictor_68_face_landmarks.dat.bz2 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    bzip2 -dk ./model_weights/shape_predictor_68_face_landmarks.dat.bz2
fi