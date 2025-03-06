import { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import UploadButton from '../../components/UploadButton/UploadButton';
import Analysis from '../../components/Analysis/Analysis';
import './Home.css';
import frameImage from '../../assets/images/cyber-crime.jpg';

const Home = () => {
  const [analysisData, setAnalysisData] = useState(null);
  const [videoPreview, setVideoPreview] = useState(null);
  const [scrollProgress, setScrollProgress] = useState(0);

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
    
    const previewUrl = URL.createObjectURL(file);
    setVideoPreview(previewUrl);

    // Simulated analysis
    const result = await new Promise(resolve => {
      setTimeout(() => {
        resolve({
          authenticity: 92.4,
          frames: Array(3).fill().map((_, i) => ({
            image: `https://picsum.photos/300/200?random=${i}`,
            artifacts: Math.floor(Math.random() * 15) + 5
          })),
          metrics: {
            styleganScore: 0.87,
            forensicMatches: 23
          }
        });
      }, 1500);
    });

    setAnalysisData(result);
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
        <div className="video-analysis-section">
          <video src={videoPreview} controls className="video-preview" />
          {analysisData && <Analysis data={analysisData} />}
        </div>
      )}
    </div>
  );
};

export default Home;
