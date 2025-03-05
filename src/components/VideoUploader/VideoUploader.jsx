import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import './VideoUploader.css';

const VideoUploader = ({ onDetection }) => {
  const [videoPreview, setVideoPreview] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleDetection = useCallback(async (file) => {
    setIsProcessing(true);
    try {
      // Simulated detection
      const result = await new Promise(resolve => setTimeout(() => ({
        isDeepfake: Math.random() > 0.5,
        confidence: (Math.random() * 100).toFixed(2),
        indicators: ['Inconsistent eye blinking', 'Unnatural facial movements']
      }), 2000));
      
      onDetection(result);
    } finally {
      setIsProcessing(false);
    }
  }, [onDetection]);

  const onDrop = useCallback(acceptedFiles => {
    const file = acceptedFiles[0];
    const previewUrl = URL.createObjectURL(file);
    setVideoPreview(previewUrl);
    handleDetection(file);
  }, [handleDetection]);

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    accept: 'video/*',
    multiple: false
  });

  return (
    <div {...getRootProps()} className="video-uploader-container glass-panel">
      <input {...getInputProps()} />
      
      {videoPreview ? (
        <div className="video-preview-container">
          <video 
            src={videoPreview} 
            controls 
            className="video-player"
          />
          {isProcessing && <div className="processing-overlay">Analyzing...</div>}
        </div>
      ) : (
        <div className="upload-prompt">
          <p className="glitch-text">DROP VIDEO TO SCAN</p>
          <p className="supported-formats">MP4, AVI, MOV</p>
        </div>
      )}
    </div>
  );
};

export default VideoUploader;