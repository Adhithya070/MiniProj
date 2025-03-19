import './UploadButton.css';

const UploadButton = ({ onFileSelect, onRemove, hasFile }) => {
  return (
    <div className={`upload-container ${hasFile ? 'has-file' : ''}`}>
      <label className="upload-button">
        <input
          type="file"
          accept="video/*"
          onChange={(e) => {
            if (e.target.files?.[0]) onFileSelect(e.target.files[0]);
          }}
          onClick={(e) => {
            if (hasFile) {
              e.preventDefault();
              onRemove();
            }
          }}
          style={{ display: 'none' }}
        />
        {hasFile ? 'REMOVE UPLOAD' : 'UPLOAD VIDEO'}
      </label>
    </div>
  );
};

export default UploadButton;
