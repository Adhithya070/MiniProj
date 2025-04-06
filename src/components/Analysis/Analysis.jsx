import { motion } from 'framer-motion';
import './Analysis.css';

const Analysis = ({ data }) => {
  return (
    <motion.div 
      className="analysis-container"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
    >
      <div className="result-banner">
        <h1>{data.overall_result.toUpperCase()}</h1>
      </div>
      <div className="details">
        {data.summary.split('\n').map((line, index) => (
          <p key={index}>{line}</p>
        ))}
      </div>
      <div className="frame-analysis">
        <h2>Video Forensic Frame Analysis</h2>
        <div className="frames-scroll">
          {data.top_frames.map((frame, i) => (
            <div key={i} className="frame-card">
              <img 
                src={`data:image/png;base64,${frame}`} 
                alt={`Frame ${i + 1}`} 
              />
            </div>
          ))}
        </div>
      </div>
      <div className="audio-analysis">
        <h2>Audio Analysis</h2>
        <p>
          Audio Fake Probability: {data.audio_result.fake_probability.toFixed(2)}
        </p>
        <p>
          Audio is classified as: {data.audio_result.fake_probability > 0.5 ? "Fake" : "Real"}
        </p>
      </div>
    </motion.div>
  );
};

export default Analysis;
