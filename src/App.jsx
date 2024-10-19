import React, { useRef, useEffect } from "react";

import "./App.css";

function App() {
  const videoRef = useRef(null);

  useEffect(() => {
    // Access the user's webcam
    const getVideo = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error("Error accessing webcam: ", err);
      }
    };

    getVideo();

    // Cleanup function to stop the stream when the component unmounts
    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        let stream = videoRef.current.srcObject;
        let tracks = stream.getTracks();
        tracks.forEach((track) => track.stop());
      }
    };
  }, []);
  return (
    <div>
      <video
        ref={videoRef}
        autoPlay
        playsInline
        style={{ width: "100%", height: "auto" }}
      >
        <track kind="captions" />
      </video>
    </div>
  );
}

export default App;
