import React, { useState } from 'react';
import {
  Container,
  Paper,
  Typography,
  TextField,
  Button,
  Box,
  Grid,
  Card,
  CardContent,
  Alert,
  CircularProgress,
  Chip,
  Divider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  IconButton,
} from '@mui/material';
import {
  Search as SearchIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  ExpandMore as ExpandMoreIcon,
  LocationOn as LocationIcon,
  Category as CategoryIcon,
  FactCheck as FactCheckIcon,
  PhotoCamera as PhotoCameraIcon,
  Delete as DeleteIcon,
} from '@mui/icons-material';
import { useMutation } from 'react-query';
import axios from 'axios';

const Analyze = () => {
  const [tweetText, setTweetText] = useState('');
  const [location, setLocation] = useState('');
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [error, setError] = useState('');

  const analyzeMutation = useMutation(
    async (data) => {
      const response = await axios.post('/api/analyze', data);
      return response.data;
    },
    {
      onSuccess: (data) => {
        setAnalysisResult(data);
        setError('');
      },
      onError: (error) => {
        setError(error.response?.data?.error || 'An error occurred during analysis');
        setAnalysisResult(null);
      },
    }
  );

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      if (file.size > 5 * 1024 * 1024) { // 5MB limit
        setError('Image size should be less than 5MB');
        return;
      }
      
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleRemoveImage = () => {
    setSelectedImage(null);
    setImagePreview(null);
  };

  const handleAnalyze = () => {
    if (!tweetText.trim()) {
      setError('Please enter tweet text to analyze');
      return;
    }

    const data = {
      text: tweetText,
      location: location.trim() || null,
      image_path: selectedImage ? selectedImage.name : null, // In real app, upload image first
    };

    analyzeMutation.mutate(data);
  };

  const getPredictionColor = (prediction) => {
    return prediction === 'real' ? 'success' : 'error';
  };

  const getDisasterColor = (disasterType) => {
    const colors = {
      wildfire: 'error',
      flood: 'info',
      hurricane: 'warning',
      earthquake: 'secondary',
    };
    return colors[disasterType] || 'default';
  };

  const formatConfidence = (confidence) => {
    return `${(confidence * 100).toFixed(1)}%`;
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom align="center">
        Fake News Detection Analysis
      </Typography>
      
      <Typography variant="body1" color="text.secondary" align="center" sx={{ mb: 4 }}>
        Enter a tweet to analyze for fake news detection and disaster classification
      </Typography>

      <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={8}>
            <TextField
              fullWidth
              multiline
              rows={4}
              label="Tweet Text"
              value={tweetText}
              onChange={(e) => setTweetText(e.target.value)}
              placeholder="Enter the tweet text to analyze..."
              variant="outlined"
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <TextField
              fullWidth
              label="Location (Optional)"
              value={location}
              onChange={(e) => setLocation(e.target.value)}
              placeholder="e.g., New York, NY"
              variant="outlined"
            />
          </Grid>
          
          {/* Image Upload Section */}
          <Grid item xs={12}>
            <Typography variant="h6" gutterBottom>
              Add Image (Optional)
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <input
                accept="image/*"
                style={{ display: 'none' }}
                id="image-upload"
                type="file"
                onChange={handleImageUpload}
              />
              <label htmlFor="image-upload">
                <Button
                  variant="outlined"
                  component="span"
                  startIcon={<PhotoCameraIcon />}
                >
                  Upload Image
                </Button>
              </label>
              {selectedImage && (
                <Button
                  variant="outlined"
                  color="error"
                  startIcon={<DeleteIcon />}
                  onClick={handleRemoveImage}
                >
                  Remove Image
                </Button>
              )}
            </Box>
            
            {imagePreview && (
              <Box sx={{ mt: 2 }}>
                <img
                  src={imagePreview}
                  alt="Preview"
                  style={{
                    maxWidth: '200px',
                    maxHeight: '200px',
                    objectFit: 'cover',
                    borderRadius: '8px',
                    border: '1px solid #ddd'
                  }}
                />
              </Box>
            )}
          </Grid>
          
          <Grid item xs={12}>
            <Button
              variant="contained"
              size="large"
              startIcon={<SearchIcon />}
              onClick={handleAnalyze}
              disabled={analyzeMutation.isLoading}
              fullWidth
            >
              {analyzeMutation.isLoading ? (
                <>
                  <CircularProgress size={20} sx={{ mr: 1 }} />
                  Analyzing...
                </>
              ) : (
                'Analyze Tweet (Multimodal)'
              )}
            </Button>
          </Grid>
        </Grid>
      </Paper>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {analysisResult && (
        <Grid container spacing={3}>
          {/* Main Results */}
          <Grid item xs={12} md={6}>
            <Card elevation={3}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Analysis Results
                </Typography>
                
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    Multimodal Analysis
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                    {analysisResult.multimodal_analysis?.is_fake ? (
                      <WarningIcon color="error" />
                    ) : (
                      <CheckCircleIcon color="success" />
                    )}
                    <Chip
                      label={analysisResult.multimodal_analysis?.prediction?.toUpperCase() || 'UNKNOWN'}
                      color={analysisResult.multimodal_analysis?.is_fake ? 'error' : 'success'}
                      variant="outlined"
                    />
                    {analysisResult.multimodal_analysis?.modality && (
                      <Chip
                        label={analysisResult.multimodal_analysis.modality}
                        color="primary"
                        size="small"
                      />
                    )}
                    <Typography variant="body2" color="text.secondary">
                      ({formatConfidence(analysisResult.multimodal_analysis?.confidence || 0)})
                    </Typography>
                  </Box>
                  <Typography variant="body2" color="text.secondary">
                    {analysisResult.multimodal_analysis?.explanation || 'No explanation available'}
                  </Typography>
                </Box>

                {analysisResult.disaster_classification.type && (
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="subtitle1" gutterBottom>
                      Disaster Classification
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                      <CategoryIcon color="primary" />
                      <Chip
                        label={analysisResult.disaster_classification.type.toUpperCase()}
                        color={getDisasterColor(analysisResult.disaster_classification.type)}
                        variant="outlined"
                      />
                      {analysisResult.disaster_classification.confidence && (
                        <Typography variant="body2" color="text.secondary">
                          ({formatConfidence(analysisResult.disaster_classification.confidence)})
                        </Typography>
                      )}
                    </Box>
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>

          {/* Fact Checking */}
          <Grid item xs={12} md={6}>
            <Card elevation={3}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Fact Checking Results
                </Typography>
                
                <Box sx={{ mb: 2 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                    <FactCheckIcon color="primary" />
                    <Chip
                      label={analysisResult.fact_checking.verified ? 'VERIFIED' : 'NOT VERIFIED'}
                      color={analysisResult.fact_checking.verified ? 'success' : 'error'}
                      variant="outlined"
                    />
                    <Typography variant="body2" color="text.secondary">
                      ({formatConfidence(analysisResult.fact_checking.confidence)})
                    </Typography>
                  </Box>
                </Box>

                {analysisResult.fact_checking.sources && analysisResult.fact_checking.sources.length > 0 && (
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Verified Sources ({analysisResult.fact_checking.sources.length})
                    </Typography>
                    {analysisResult.fact_checking.sources.slice(0, 3).map((source, index) => (
                      <Typography key={index} variant="body2" color="text.secondary" sx={{ mb: 0.5 }}>
                        • {source.source}: {source.title}
                      </Typography>
                    ))}
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>

          {/* Detailed Analysis */}
          <Grid item xs={12}>
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="h6">Detailed Analysis</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle1" gutterBottom>
                      Original Text
                    </Typography>
                    <Paper variant="outlined" sx={{ p: 2, bgcolor: 'grey.50' }}>
                      <Typography variant="body2">{analysisResult.text}</Typography>
                    </Paper>
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle1" gutterBottom>
                      Processed Text
                    </Typography>
                    <Paper variant="outlined" sx={{ p: 2, bgcolor: 'grey.50' }}>
                      <Typography variant="body2">{analysisResult.processed_text}</Typography>
                    </Paper>
                  </Grid>

                  {analysisResult.location_info && (
                    <Grid item xs={12}>
                      <Typography variant="subtitle1" gutterBottom>
                        Location Information
                      </Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <LocationIcon color="primary" />
                        <Typography variant="body2">
                          {analysisResult.location_info.address || 'Location not available'}
                        </Typography>
                      </Box>
                    </Grid>
                  )}

                  {analysisResult.fact_checking.explanations && (
                    <Grid item xs={12}>
                      <Typography variant="subtitle1" gutterBottom>
                        Fact Checking Explanations
                      </Typography>
                      {analysisResult.fact_checking.explanations.map((explanation, index) => (
                        <Typography key={index} variant="body2" color="text.secondary" sx={{ mb: 0.5 }}>
                          • {explanation}
                        </Typography>
                      ))}
                    </Grid>
                  )}
                </Grid>
              </AccordionDetails>
            </Accordion>
          </Grid>
        </Grid>
      )}
    </Container>
  );
};

export default Analyze; 