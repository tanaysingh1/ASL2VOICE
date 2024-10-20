import React, { useRef, useEffect, useState } from "react";
import * as handpose from "@tensorflow-models/handpose";
import * as tf from "@tensorflow/tfjs";
import "./App.css";

function App() {
  const webcamRef = useRef(null);
  const [model, setModel] = useState(null);
  const [landmarks, setLandmarks] = useState([]);
  const [dataset, setDataset] = useState([]);
  const [labels, setLabels] = useState([]);
  const [detectorModel, setDetectorModel] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [isModelReady, setIsModelReady] = useState(false);
  const [detecting, setDetecting] = useState(false);
  const [input, setInput] = useState("");
  useEffect(() => {
    const loadHandposeModel = async () => {
      const handposeModel = await handpose.load();
      setModel(handposeModel);
      setIsModelReady(true);
      console.log("Handpose model loaded.");
    };

    const setupWebcam = async () => {
      const video = webcamRef.current;
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      video.srcObject = stream;
      video.play();
    };

    loadHandposeModel();
    setupWebcam();
  }, []);

  useEffect(() => {
    if (isModelReady && webcamRef.current) {
      detectHand();
    }
  }, [isModelReady, webcamRef]);

  // useEffect(() => {
  //   if (detectorModel) {
  //     const intervalId = setInterval(() => predictSign(), 1000);
  //     return () => clearInterval(intervalId);
  //   }
  // }, [detectorModel]);

  const detectHand = async () => {
    if (!webcamRef.current || !model) return;

    const video = webcamRef.current;
    const predictions = await model.estimateHands(video);
    setDetecting(true);

    if (predictions.length > 0) {
      const flatLandmarks = predictions[0].landmarks.flat();
      setLandmarks(flatLandmarks);
      console.log("Hand detected, landmarks:", flatLandmarks.length);
    } else {
      setLandmarks([]);
      console.log("No hand detected.");
    }

    requestAnimationFrame(detectHand);
  };

  const captureSign = async (label) => {
    if (!detecting) {
      await detectHand();
    }
    if (landmarks.length === 63) {
      setDataset((prevDataset) => [...prevDataset, landmarks]);
      setLabels((prevLabels) => [...prevLabels, label]);
      console.log(`Captured label: ${label}`, landmarks);
    } else {
      console.warn("Landmarks are incomplete, not adding to dataset.");
    }
  };

  const trainModel = async () => {
    if (dataset.length === 0 || labels.length === 0) {
      console.error("No data to train on!");
      return;
    }

    console.log("Training started...");

    // Convert dataset and labels into tensors
    const xs = tf.tensor2d(dataset);
    const ys = tf.oneHot(
      labels.map((l) => l.charCodeAt(0) - 65),
      labels.length
    ); // Adjust if needed for more classes

    // Build a more complex model
    const model = tf.sequential();
    model.add(
      tf.layers.dense({ inputShape: [63], units: 256, activation: "relu" })
    );
    model.add(
      tf.layers.dense({ inputShape: [63], units: 512, activation: "relu" })
    );
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.dropout({ rate: 0.8 }));

    model.add(tf.layers.dense({ units: 256, activation: "relu" }));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.dropout({ rate: 0.8 }));

    model.add(tf.layers.dense({ units: 128, activation: "relu" }));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.dropout({ rate: 0.8 }));

    // Output layer
    model.add(tf.layers.dense({ units: labels.length, activation: "softmax" }));

    // Compile the model with categoricalCrossentropy loss for multi-class classification
    model.compile({
      optimizer: tf.train.adam(0.0001), // Start with a higher learning rate
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"],
    });

    // Train the model
    await model.fit(xs, ys, {
      epochs: 100,
      batchSize: 16,
      validationSplit: 0.2, // Add validation split to monitor overfitting
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          console.log(
            `Epoch ${epoch + 1}: Loss = ${logs.loss}, Accuracy = ${logs.acc}`
          );
        },
      },
    });

    setDetectorModel(model);
    console.log("Training complete.");
  };

  const predictSign = () => {
    if (!detectorModel || landmarks.length !== 63) return;

    const video = webcamRef.current;
    console.log(landmarks);
    // Normalize the landmarks
    // const normalizedLandmarks = landmarks.map(
    //   (val, index) =>
    //     val / (index % 3 === 0 ? video.videoWidth : video.videoHeight)
    // );
    // Predict the sign using the trained model
    const inputTensor = tf.tensor2d([landmarks]);
    const predictionTensor = detectorModel.predict(inputTensor);
    const predictedIndex = predictionTensor.argMax(-1).dataSync()[0];
    const predictedLabel = String.fromCharCode(predictedIndex + 65); // Adjust for multi-class prediction
    console.log(predictedIndex);
    setPrediction(predictedLabel);
    console.log("Prediction:", predictedLabel);

    inputTensor.dispose();
    predictionTensor.dispose();
  };

  return (
    <div>
      <h1>Sign Language Recognition</h1>
      <video
        ref={webcamRef}
        autoPlay
        playsInline
        style={{ width: "100%", height: "auto" }}
      />
      <div>
        <input value={input} onChange={(e) => setInput(e.target.value)}></input>
        <button onClick={() => captureSign(input)}>Capture {input}</button>
        <button onClick={() => captureSign("B")}>Capture "B"</button>
      </div>
      <button onClick={trainModel}>Train Model</button>
      <button onClick={predictSign}>Predict Now</button>
      <h2>Prediction: {prediction ? prediction : "Detecting..."}</h2>
      <p>Landmarks detected: {landmarks.length}</p>
      <p>Dataset size: {dataset.length}</p>
      <p>Model trained: {detectorModel ? "Yes" : "No"}</p>
    </div>
  );
}

export default App;
