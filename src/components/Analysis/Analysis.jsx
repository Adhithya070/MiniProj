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
        <h1>{data.is_deepfake ? "DEEPFAKE" : "AUTHENTIC"}</h1>
      </div>
      <div className="details">
        {data.summary.split('\n').map((line, index) => (
          <p key={index}>{line}</p>
        ))}
      </div>
      <div className="frame-analysis">
        <h2>Forensic Frame Analysis</h2>
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
    </motion.div>
  );
};

export default Analysis;
