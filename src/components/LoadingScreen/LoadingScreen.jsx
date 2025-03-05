// components/LoadingScreen/LoadingScreen.jsx
import { useEffect, useState } from 'react';
import './LoadingScreen.css';

const LoadingScreen = ({ onLoaded }) => {
  const [loaded, setLoaded] = useState(false);
  const letters = "DeepGuard".split("");

  useEffect(() => {
    const timer = setTimeout(() => {
      setLoaded(true);
      onLoaded();
    }, 3000);

    return () => clearTimeout(timer);
  }, [onLoaded]);

  return (
    <div className={`loading-screen ${loaded ? 'loaded' : ''}`}>
      <div className="vortax-text">
        {letters.map((letter, index) => (
          <span 
            key={index}
            className="letter-fade"
            style={{ animationDelay: `${index * 0.2}s` }}
          >
            {letter}
          </span>
        ))}
      </div>
    </div>
  );
};

export default LoadingScreen;