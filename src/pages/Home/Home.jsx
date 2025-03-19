import { useState, useEffect, useCallback } from 'react';
import UploadButton from '../../components/UploadButton/UploadButton';
import Analysis from '../../components/Analysis/Analysis';
import './Home.css';
import frameImage from '../../assets/images/cyber-crime.jpg';

const Home = () => {
  const [analysisData, setAnalysisData] = useState(null);
  const [videoPreview, setVideoPreview] = useState(null);
  const [scrollProgress, setScrollProgress] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleScroll = useCallback(() => {
    const scrollY = window.scrollY;
    const windowHeight = window.innerHeight;
    const progress = Math.min(scrollY / windowHeight, 1);
    setScrollProgress(progress);
  }, []);

  useEffect(() => {
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, [handleScroll]);

  const handleAnalysis = useCallback(async (file) => {
    if (!file) return;

    setIsProcessing(true);
    const previewUrl = URL.createObjectURL(file);
    setVideoPreview(previewUrl);

    const formData = new FormData();
    formData.append('video', file);

    try {
      const response = await fetch('http://localhost:5000/analyze', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        throw new Error('Analysis failed');
      }
      const result = await response.json();
      setAnalysisData(result);
    } catch (error) {
      console.error('Error during analysis:', error);
    } finally {
      setIsProcessing(false);
    }
  }, []);

  const handleRemove = useCallback(() => {
    setVideoPreview(null);
    setAnalysisData(null);
  }, []);

  return (
    <div className="home-container">
      <section className="hero-section">
        <div className="title-frame">
          <h1 className="main-title">Deepfake Detection</h1>
          <div className="image-container">
            <img 
              src={frameImage} 
              alt="Detection System" 
              className="frame-image"
              style={{
                transform: `scale(${1 + scrollProgress * 0.5})`,
                opacity: 1 - scrollProgress
              }}
            />
          </div>
          <p className="project-description">
            ADVANCED AI-POWERED DETECTION SYSTEM<br/>
            UNMASKING DIGITAL DECEPTION
          </p>
          <UploadButton 
            onFileSelect={handleAnalysis}
            onRemove={handleRemove}
            hasFile={!!videoPreview}
          />
        </div>
      </section>

      {/* Analysis Section */}
      {videoPreview && (
        <div className="video-analysis-section" style={{ position: 'relative' }}>
          <video src={videoPreview} controls className="video-preview" />
          {isProcessing && (
            <div className="processing-overlay">
              <div className="spinner"></div>
              <div>Analyzing...</div>
            </div>
          )}
          {analysisData && <Analysis data={analysisData} />}
        </div>
      )}
    </div>
  );
};

export default Home;
