import { useState } from 'react'
import './App.css'

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)
  const [dragOver, setDragOver] = useState(false)

  const handleFileSelect = (file) => {
    if (file && file.type === 'audio/mpeg') {
      setSelectedFile(file)
      setError(null)
    } else {
      setError('Please upload a valid MP3 file')
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    handleFileSelect(file)
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setDragOver(true)
  }

  const handleDragLeave = () => {
    setDragOver(false)
  }

  const handleFileInputChange = (e) => {
    const file = e.target.files[0]
    handleFileSelect(file)
  }

  const pollResults = async (uploadId) => {
    const maxAttempts = 60 // Poll for up to 60 seconds
    let attempts = 0

    const poll = async () => {
      try {
        const response = await fetch(`/api/results/${uploadId}`)
        
        if (!response.ok) {
          throw new Error('Failed to fetch results')
        }

        const data = await response.json()

        if (data.status === 'completed') {
          setResults(data.results || [])
          setLoading(false)
        } else if (data.status === 'failed') {
          throw new Error(data.error || 'Processing failed')
        } else if (data.status === 'processing') {
          attempts++
          if (attempts < maxAttempts) {
            setTimeout(poll, 1000) // Poll every second
          } else {
            throw new Error('Processing timeout')
          }
        }
      } catch (err) {
        setError(err.message)
        setLoading(false)
      }
    }

    poll()
  }

  const handleUpload = async () => {
    if (!selectedFile) return

    setLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      const response = await fetch('/api/upload-and-search', {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Upload failed')
      }

      const data = await response.json()
      
      // Start polling for results
      if (data.upload_id) {
        pollResults(data.upload_id)
      } else {
        throw new Error('No upload ID returned')
      }
    } catch (err) {
      setError(err.message)
      setLoading(false)
    }
  }

  const handleReset = () => {
    setSelectedFile(null)
    setResults(null)
    setError(null)
  }

  if (loading) {
    return (
      <div className="container">
        <div className="loading">
          <div className="spinner"></div>
          <p>Analyzing your song...</p>
          <p className="loading-subtitle">This may take a moment</p>
        </div>
      </div>
    )
  }

  if (results) {
    return (
      <div className="container">
        <h1>🎵 Similar Songs</h1>
        <div className="results">
          {results.length === 0 ? (
            <p style={{ textAlign: 'center', color: '#666' }}>No similar tracks found</p>
          ) : (
            results.map((track, index) => (
              <div key={index} className="result-item">
                <div className="track-info">
                  <div className="track-title">{track.title || `Track ${track.track_id}`}</div>
                  <div className="track-id">Track ID: {track.track_id}</div>
                </div>
                <div className="similarity">
                  #{track.order + 1}
                </div>
              </div>
            ))
          )}
        </div>
        <button className="back-btn" onClick={handleReset}>
          Upload Another Song
        </button>
      </div>
    )
  }

  return (
    <div className="container">
      <h1>🎵 Seamless Playlist Generator</h1>
      
      <div
        className={`upload-area ${dragOver ? 'dragover' : ''}`}
        onClick={() => document.getElementById('fileInput').click()}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        <div className="upload-icon">📁</div>
        <p>Drag & drop your MP3 file here</p>
        <p className="upload-subtitle">or click to browse</p>
      </div>

      <input
        type="file"
        id="fileInput"
        accept=".mp3,audio/mpeg"
        onChange={handleFileInputChange}
        style={{ display: 'none' }}
      />

      {selectedFile && (
        <div className="file-name">
          Selected: {selectedFile.name}
        </div>
      )}

      {selectedFile && (
        <button className="upload-btn" onClick={handleUpload}>
          Find Similar Songs
        </button>
      )}

      {error && (
        <div className="error">
          {error}
        </div>
      )}
    </div>
  )
}

export default App
