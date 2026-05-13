import { useState } from 'react'
import './App.css'

function App() {
  const [selectedImage, setSelectedImage] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [resultData, setResultData] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [activeModel, setActiveModel] = useState(null)
  const [error, setError] = useState(null)
  const [backendUrl, setBackendUrl] = useState('https://abhisam-underwater-waste-detection.hf.space')
  const [showSettings, setShowSettings] = useState(false)

  const handleImageChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      setSelectedImage(file)
      setPreviewUrl(URL.createObjectURL(file))
      setResultData(null)
      setError(null)
    }
  }

  const handleUpload = async (version) => {
    if (!selectedImage) return
    setIsLoading(true)
    setActiveModel(version)
    setError(null)

    const formData = new FormData()
    formData.append('image', selectedImage)
    formData.append('version', version)

    try {
      const response = await fetch(`${backendUrl}/predict`, {
        method: 'POST',
        headers: {
          'Bypass-Tunnel-Reminder': 'true'
        },
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `Server responded with ${response.status}: ${response.statusText}`);
      }

      const data = await response.json()
      setResultData(data)
    } catch (err) {
      console.error('Detection Error:', err);
      if (err.name === 'TypeError' && err.message === 'Failed to fetch') {
        setError('Connection failed. Please ensure the backend is running and the tunnel URL in App.jsx is current.');
      } else {
        setError(err.message);
      }
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="App">
      <header>
        <h1>Underwater Waste Vision</h1>
        <p style={{ opacity: 0.7, marginBottom: '0.5rem' }}>Multi-Model Comparative Analysis</p>
        
        <div style={{ marginBottom: '1.5rem' }}>
          <button 
            className="premium-button"
            style={{ 
              background: 'rgba(255,255,255,0.1)', 
              fontSize: '0.8rem', 
              padding: '0.4rem 1rem',
              border: '1px solid rgba(255,255,255,0.2)'
            }}
            onClick={() => setShowSettings(!showSettings)}
          >
            {showSettings ? '✕ Close Settings' : '⚙️ Connection Settings'}
          </button>
        </div>

        {showSettings && (
          <div style={{ 
            marginBottom: '2rem', 
            padding: '1.5rem', 
            background: 'rgba(255,255,255,0.05)', 
            borderRadius: '12px',
            border: '1px solid rgba(255,255,255,0.1)',
            textAlign: 'left'
          }}>
            <label style={{ display: 'block', marginBottom: '0.8rem', fontWeight: '600', fontSize: '0.9rem' }}>
              Backend Tunnel URL:
            </label>
            <input 
              type="text" 
              value={backendUrl} 
              onChange={(e) => setBackendUrl(e.target.value)}
              placeholder="https://your-tunnel.loca.lt"
              style={{ 
                width: '100%', 
                padding: '0.8rem', 
                borderRadius: '8px', 
                border: '1px solid rgba(255,255,255,0.2)', 
                background: 'rgba(0,0,0,0.3)', 
                color: 'white',
                fontSize: '1rem'
              }}
            />
            <p style={{ fontSize: '0.75rem', marginTop: '0.8rem', opacity: 0.6 }}>
              Update this URL if the connection fails or the tunnel expires.
            </p>
          </div>
        )}
      </header>

      <main className="glass-card">
        {!resultData ? (
          <div className="upload-section">
            <div className="upload-zone" onClick={() => document.getElementById('fileInput').click()}>
              {previewUrl ? (
                <img src={previewUrl} alt="Preview" style={{ maxWidth: '100%', maxHeight: '400px', borderRadius: '8px' }} />
              ) : (
                <div className="upload-placeholder">
                  <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>📷</div>
                  <p>Click or Drag to Upload Underwater Image</p>
                </div>
              )}
              <input
                id="fileInput"
                type="file"
                accept="image/*"
                onChange={handleImageChange}
                style={{ display: 'none' }}
              />
            </div>

            <div className="model-selector">
              <button
                className={`premium-button ${isLoading && activeModel === 'yolov9' ? 'loading' : ''}`}
                onClick={() => handleUpload('yolov9')}
                disabled={!selectedImage || isLoading}
              >
                {isLoading && activeModel === 'yolov9' ? 'Processing...' : 'Test YOLOv9'}
              </button>
              <button
                className={`premium-button ${isLoading && activeModel === 'yolov10' ? 'loading' : ''}`}
                onClick={() => handleUpload('yolov10')}
                disabled={!selectedImage || isLoading}
              >
                {isLoading && activeModel === 'yolov10' ? 'Processing...' : 'Test YOLOv10'}
              </button>
              <button
                className={`premium-button ${isLoading && activeModel === 'yolov11' ? 'loading' : ''}`}
                onClick={() => handleUpload('yolov11')}
                disabled={!selectedImage || isLoading}
              >
                {isLoading && activeModel === 'yolov11' ? 'Processing...' : 'Test YOLOv11'}
              </button>
            </div>
            {error && <p style={{ color: '#ef4444', marginTop: '1rem' }}>{error}</p>}
          </div>
        ) : (
          <div className="results-section">
            <div className="result-header">
              <div className="badge">{activeModel.toUpperCase()} Results</div>
              <h3>Detection Summary</h3>
              <p style={{ fontSize: '0.9rem', opacity: 0.8, marginBottom: '0.5rem' }}>
                Resolution: {resultData.image_size.width} x {resultData.image_size.height}
              </p>
              <p>
                Found <strong>{resultData.count}</strong> items
                {resultData.count > 0 && (
                  <span>: <strong>{
                    Object.entries(
                      resultData.detections.reduce((acc, curr) => {
                        acc[curr.class] = (acc[curr.class] || 0) + 1;
                        return acc;
                      }, {})
                    )
                      .map(([cls, count]) => `${count} ${cls}${count > 1 ? 's' : ''}`)
                      .join(', ')
                  }</strong></span>
                )}
              </p>
              <button
                className="premium-button"
                style={{ marginBottom: '2rem' }}
                onClick={() => {
                  setResultData(null)
                  setPreviewUrl(null)
                  setSelectedImage(null)
                }}
              >
                Upload New Image
              </button>
            </div>

            <div className="result-container">
              <div>
                <h4>Original Image</h4>
                <img src={previewUrl} alt="Original" className="result-image" />
              </div>
              <div>
                <h4>Detections</h4>
                <img
                  src={`data:image/jpeg;base64,${resultData.result_image_b64}`}
                  alt="Detected"
                  className="result-image"
                />
              </div>
            </div>
          </div>
        )}
      </main>

      <footer style={{ marginTop: '3rem', opacity: 0.5, fontSize: '0.9rem' }}>
        &copy; 2024 Underwater Waste Detection System - AI for Planet
      </footer>
    </div>
  )
}

export default App
