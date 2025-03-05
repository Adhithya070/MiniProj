import './About.css';

const About = () => {
  return (
    <div className="about-container">
      <h1>About Our Technology</h1>
      
      <div className="content-section">
        <h2>Core Architecture</h2>
        <p>
          Our system leverages StyleGAN2 trained on the Forensic++ dataset to detect
          subtle artifacts in generated media. The model analyzes both spatial
          and temporal inconsistencies at multiple resolution levels.
        </p>
      </div>

      <div className="content-section">
        <h2>Key Features</h2>
        <ul className="features-list">
          <li>Multi-scale texture analysis</li>
          <li>Temporal consistency checking</li>
          <li>Frequency domain artifacts detection</li>
          <li>Ensemble verification system</li>
        </ul>
      </div>

      <div className="content-section">
        <h2>Technical Specifications</h2>
        <div className="specs-grid">
          <div className="spec-card">
            <h3>Model Architecture</h3>
            <p>Modified StyleGAN2 with attention mechanisms</p>
          </div>
          <div className="spec-card">
            <h3>Training Data</h3>
            <p>Forensic++ Dataset (1.2M labeled samples)</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default About;