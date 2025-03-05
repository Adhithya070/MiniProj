import { motion } from 'framer-motion';
import './Analysis.css';

const Analysis = ({ data }) => {
  return (
    <motion.div 
      className="analysis-container"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
    >
      <div className="metrics">
        <div className="metric-card">
          <h3>Authenticity Score</h3>
          <div className="metric-value">{data.authenticity}%</div>
        </div>
        <div className="metric-card">
          <h3>StyleGAN Confidence</h3>
          <div className="metric-value">{data.metrics.styleganScore}</div>
        </div>
      </div>

      <div className="frame-analysis">
        <h2>Forensic Frame Analysis</h2>
        <div className="frames-grid">
          {data.frames.map((frame, i) => (
            <div key={i} className="frame-card">
              <img src={frame.image} alt={`Frame ${i+1}`} />
              <div className="frame-details">
                <span>Artifacts Detected: {frame.artifacts}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </motion.div>
  );
};

export default Analysis;