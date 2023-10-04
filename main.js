// Copyright 2023 The MediaPipe Authors.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
import {
  HandLandmarker,
  FaceLandmarker,
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils,
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3';

let handLandmarker = undefined;
let faceLandmarker = undefined;
let poseLandmarker = undefined;
let handLandmarkerResult = undefined;
let faceLandmarkerResult = undefined;
let poseLandmarkerResult = undefined;
let webcamRunning = false;
let lastVideoTime = -1;
const videoWidth = 480;
const video = document.getElementById('webcam');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');
const drawingUtils = new DrawingUtils(canvasCtx);
const enableWebcamButton = document.getElementById('webcamButton');

// Enable the live webcam view and start detection.
const enableCam = (event) => {
  if (!handLandmarker || !faceLandmarker || !poseLandmarker) {
    console.log('Wait! mediapipeTasks not loaded yet.');
    return;
  }

  if (webcamRunning === true) {
    webcamRunning = false;
    enableWebcamButton.innerText = 'ENABLE PREDICTIONS';
  } else {
    webcamRunning = true;
    enableWebcamButton.innerText = 'DISABLE PREDICTIONS';
  }

  // Activate the webcam stream.
  navigator.mediaDevices
    .getUserMedia({ video: true })
    .then((stream) => {
      video.srcObject = stream;
      video.addEventListener('loadeddata', predictWebcam);
    })
    .catch((err) => {
      console.error(`${err.name}: ${err.message}`);
    });
};

// If webcam supported, add event listener to button for when user wants to activate it.
if (!!navigator.mediaDevices.getUserMedia) {
  enableWebcamButton.addEventListener('click', enableCam);
} else {
  console.warn('getUserMedia() is not supported.');
}

// Before we can use HandLandmarker class we must wait for it to finish loading.
// Machine Learning models can be large and take a moment to get everything needed to run.
const createMediapipeTasks = async () => {
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
    numHands: 2,
  });
  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
    outputFaceBlendshapes: true,
  });
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
    numPoses: 1,
  });
};
createMediapipeTasks();

const predictWebcam = async () => {
  const radio = video.videoHeight / video.videoWidth;
  video.style.width = videoWidth + 'px';
  video.style.height = videoWidth * radio + 'px';
  canvasElement.style.width = videoWidth + 'px';
  canvasElement.style.height = videoWidth * radio + 'px';
  canvasElement.width = video.videoWidth;
  canvasElement.height = video.videoHeight;

  // Now let's start detecting the stream.
  let startTimeMs = performance.now();
  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;
    handLandmarkerResult = handLandmarker.detectForVideo(video, startTimeMs);
    faceLandmarkerResult = faceLandmarker.detectForVideo(video, startTimeMs);
    poseLandmarkerResult = poseLandmarker.detectForVideo(video, startTimeMs);
  }

  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  if (handLandmarkerResult.landmarks) {
    for (const landmarks of handLandmarkerResult.landmarks) {
      drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, {
        color: '#00FF00',
        lineWidth: 5,
      });
      drawingUtils.drawLandmarks(landmarks, { color: '#FF0000', lineWidth: 2 });
    }
  }
  if (faceLandmarkerResult.faceLandmarks) {
    for (const landmarks of faceLandmarkerResult.faceLandmarks) {
      drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, {
        color: '#C0C0C070',
        lineWidth: 1,
      });
      drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, { color: '#FF3030' });
      drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, { color: '#FF3030' });
      drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, { color: '#30FF30' });
      drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, { color: '#30FF30' });
      drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: '#E0E0E0' });
      drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, { color: '#E0E0E0' });
      drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS, { color: '#FF3030' });
      drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS, { color: '#30FF30' });
    }
  }
  if (poseLandmarkerResult.landmarks) {
    for (const landmarks of poseLandmarkerResult.landmarks) {
      drawingUtils.drawLandmarks(landmarks, {
        radius: (data) => DrawingUtils.lerp(data.from.z, -0.15, 0.1, 5, 1),
      });
      drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS);
    }
  }
  canvasCtx.restore();

  //
  // console.log(handLandmarkerResult);
  // console.log(faceLandmarkerResult);
  // console.log(poseLandmarkerResult);

  updateWorldLandmarksTrace(handLandmarkerResult, poseLandmarkerResult);

  // Call this function again to keep predicting when the browser is ready.
  if (webcamRunning === true) {
    window.requestAnimationFrame(predictWebcam);
  }
};

// Plot worldLandmarker as 3D graphs with Plotly.
let leftHandLandmarksTrace = {
  x: [],
  y: [],
  z: [],
  mode: 'markers',
  marker: {
    size: 3,
    opacity: 0.8,
  },
  type: 'scatter3d',
};
let rightHandLandmarksTrace = {
  x: [],
  y: [],
  z: [],
  mode: 'markers',
  marker: {
    size: 3,
    opacity: 0.8,
  },
  type: 'scatter3d',
};
let poseLandmarksTrace = {
  x: [],
  y: [],
  z: [],
  mode: 'markers',
  marker: {
    size: 3,
    opacity: 0.8,
  },
  type: 'scatter3d',
};
let HandLandmarkerLayout = {
  scene: {
    aspectratio: {
      x: 1,
      y: 0.5,
      z: 0.5,
    },
    xaxis: {
      range: [-0.4, 0.4],
    },
    yaxis: {
      range: [-0.2, 0.2],
    },
    zaxis: {
      range: [-0.2, 0.2],
    },
    camera: {
      eye: { x: 0, y: 0, z: -1.5 },
      center: { x: 0, y: 0, z: 0 },
      up: { x: 0, y: -1, z: 0 },
    },
  },
  showlegend: false,
  staticPlot: true,
};
let PoseLandmarkerLayout = {
  scene: {
    aspectratio: {
      x: 1,
      y: 1,
      z: 1,
    },
    xaxis: {
      range: [-1, 1],
    },
    yaxis: {
      range: [-1, 1],
    },
    zaxis: {
      range: [-1, 1],
    },
    camera: {
      eye: { x: 0, y: 0, z: -1.5 },
      center: { x: 0, y: 0, z: 0 },
      up: { x: 0, y: -1, z: 0 },
    },
  },
  showlegend: false,
  staticPlot: true,
};
Plotly.newPlot('3dHandView', [leftHandLandmarksTrace, rightHandLandmarksTrace], HandLandmarkerLayout);
Plotly.newPlot('3dPoseView', [poseLandmarksTrace], PoseLandmarkerLayout);

const updateWorldLandmarksTrace = (handLandmarkerResult, poseLandmarkerResult) => {
  leftHandLandmarksTrace.y = [];
  leftHandLandmarksTrace.x = [];
  leftHandLandmarksTrace.z = [];
  rightHandLandmarksTrace.y = [];
  rightHandLandmarksTrace.x = [];
  rightHandLandmarksTrace.z = [];
  poseLandmarksTrace.x = [];
  poseLandmarksTrace.y = [];
  poseLandmarksTrace.z = [];

  if (poseLandmarkerResult.worldLandmarks.length) {
    poseLandmarksTrace.x = poseLandmarkerResult.worldLandmarks[0].map((elm) => {
      return elm.x * -1;
    });
    poseLandmarksTrace.y = poseLandmarkerResult.worldLandmarks[0].map((elm) => {
      return elm.y;
    });
    poseLandmarksTrace.z = poseLandmarkerResult.worldLandmarks[0].map((elm) => {
      return elm.z;
    });
  }
  for (let i in handLandmarkerResult.worldLandmarks) {
    if (handLandmarkerResult.handednesses[i][0].categoryName == 'Left') {
      leftHandLandmarksTrace.x = handLandmarkerResult.worldLandmarks[i].map((elm) => {
        return elm.x * -1 - 0.2;
      });
      leftHandLandmarksTrace.y = handLandmarkerResult.worldLandmarks[i].map((elm) => {
        return elm.y;
      });
      leftHandLandmarksTrace.z = handLandmarkerResult.worldLandmarks[i].map((elm) => {
        return elm.z;
      });
    } else {
      rightHandLandmarksTrace.x = handLandmarkerResult.worldLandmarks[i].map((elm) => {
        return elm.x * -1 + 0.2;
      });
      rightHandLandmarksTrace.y = handLandmarkerResult.worldLandmarks[i].map((elm) => {
        return elm.y;
      });
      rightHandLandmarksTrace.z = handLandmarkerResult.worldLandmarks[i].map((elm) => {
        return elm.z;
      });
    }
  }

  Plotly.redraw('3dHandView');
  Plotly.redraw('3dPoseView');
};
