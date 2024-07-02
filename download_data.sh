if [ ! -d "./code/data" ]; then
  tar -xvzf './landmarks_task.tgz'
  mv ./landmarks_task ./code/data
fi

if [ ! -d "./code/model_weights" ]; then
    mkdir -p ./code/model_weights
    wget -O ./code/model_weights/mmod_human_face_detector.dat.bz2 http://dlib.net/files/mmod_human_face_detector.dat.bz2 
    bzip2 -dk ./code/model_weights/mmod_human_face_detector.dat.bz2

    wget -O ./code/model_weights/shape_predictor_68_face_landmarks.dat.bz2 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    bzip2 -dk ./code/model_weights/shape_predictor_68_face_landmarks.dat.bz2
fi