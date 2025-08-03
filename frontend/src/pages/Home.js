import React from 'react';
import {
  Container,
  Typography,
  Box,
  Grid,
  Card,
  CardContent,
  Button,
  Paper,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
} from '@mui/material';
import {
  Security as SecurityIcon,
  Speed as SpeedIcon,
  CheckCircle as AccuracyIcon,
  LocationOn as LocationIcon,
  FactCheck as FactCheckIcon,
  ContactSupport as ContactIcon,
  TrendingUp as TrendingIcon,
  Psychology as PsychologyIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

const Home = () => {
  const navigate = useNavigate();

  const features = [
    {
      icon: <SecurityIcon color="primary" />,
      title: 'Fake News Detection',
      description: 'Advanced AI-powered classification to distinguish between real and fake disaster reports.',
    },
    {
      icon: <AccuracyIcon color="primary" />,
      title: 'Disaster Classification',
      description: 'Automatically categorize real disasters into wildfire, flood, hurricane, or earthquake.',
    },
    {
      icon: <FactCheckIcon color="primary" />,
      title: 'Fact Checking',
      description: 'Verify authenticity through multiple sources and credibility indicators.',
    },
    {
      icon: <LocationIcon color="primary" />,
      title: 'Location-Based Services',
      description: 'Identify relevant authorities based on tweet location and disaster type.',
    },
    {
      icon: <ContactIcon color="primary" />,
      title: 'Authority Contact',
      description: 'Direct communication with emergency services for verified disasters.',
    },
    {
      icon: <SpeedIcon color="primary" />,
      title: 'Real-time Processing',
      description: 'Live analysis of incoming disaster reports with immediate results.',
    },
  ];

  const benefits = [
    'Reduces spread of misinformation during critical times',
    'Improves emergency response coordination',
    'Provides verified information to authorities',
    'Helps maintain public safety and trust',
    'Supports disaster management efforts',
    'Enables faster emergency response times',
  ];

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Hero Section */}
      <Box sx={{ textAlign: 'center', mb: 6 }}>
        <Typography variant="h2" component="h1" gutterBottom sx={{ fontWeight: 'bold' }}>
          Fake News Detection
        </Typography>
        <Typography variant="h4" component="h2" gutterBottom color="primary">
          During Natural Disasters
        </Typography>
        <Typography variant="h6" color="text.secondary" sx={{ mb: 4, maxWidth: 800, mx: 'auto' }}>
          A comprehensive multimodal system that detects fake news and classifies real disasters,
          helping authorities respond effectively during critical times.
        </Typography>
        <Button
          variant="contained"
          size="large"
          onClick={() => navigate('/analyze')}
          sx={{ mr: 2 }}
        >
          Start Analysis
        </Button>
        <Button
          variant="outlined"
          size="large"
          onClick={() => navigate('/about')}
        >
          Learn More
        </Button>
      </Box>

      {/* Features Section */}
      <Box sx={{ mb: 6 }}>
        <Typography variant="h4" component="h2" gutterBottom align="center" sx={{ mb: 4 }}>
          Key Features
        </Typography>
        <Grid container spacing={3}>
          {features.map((feature, index) => (
            <Grid item xs={12} sm={6} md={4} key={index}>
              <Card elevation={2} sx={{ height: '100%' }}>
                <CardContent sx={{ textAlign: 'center', p: 3 }}>
                  <Box sx={{ mb: 2 }}>
                    {feature.icon}
                  </Box>
                  <Typography variant="h6" component="h3" gutterBottom>
                    {feature.title}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {feature.description}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Box>

      {/* How It Works Section */}
      <Box sx={{ mb: 6 }}>
        <Typography variant="h4" component="h2" gutterBottom align="center" sx={{ mb: 4 }}>
          How It Works
        </Typography>
        <Grid container spacing={4} alignItems="center">
          <Grid item xs={12} md={6}>
            <Paper elevation={3} sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Step 1: Input Analysis
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                Users input disaster-related tweets or text content for analysis.
              </Typography>
              
              <Typography variant="h6" gutterBottom>
                Step 2: AI Processing
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                Advanced machine learning models analyze the content for authenticity and classify disaster types.
              </Typography>
              
              <Typography variant="h6" gutterBottom>
                Step 3: Fact Checking
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                The system verifies information through multiple sources and credibility indicators.
              </Typography>
              
              <Typography variant="h6" gutterBottom>
                Step 4: Authority Contact
              </Typography>
              <Typography variant="body2" color="text.secondary">
                For verified real disasters, the system identifies and contacts relevant authorities.
              </Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} md={6}>
            <Box sx={{ textAlign: 'center' }}>
              <PsychologyIcon sx={{ fontSize: 120, color: 'primary.main', mb: 2 }} />
              <Typography variant="h6" gutterBottom>
                Multimodal Analysis
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Our system combines text analysis, image processing, and fact-checking
                to provide comprehensive fake news detection during natural disasters.
              </Typography>
            </Box>
          </Grid>
        </Grid>
      </Box>

      {/* Benefits Section */}
      <Box sx={{ mb: 6 }}>
        <Typography variant="h4" component="h2" gutterBottom align="center" sx={{ mb: 4 }}>
          Benefits
        </Typography>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Paper elevation={2} sx={{ p: 3, height: '100%' }}>
              <Typography variant="h6" gutterBottom>
                For Emergency Services
              </Typography>
              <List>
                {benefits.slice(0, 3).map((benefit, index) => (
                  <ListItem key={index} sx={{ py: 0.5 }}>
                    <ListItemIcon>
                      <TrendingIcon color="primary" />
                    </ListItemIcon>
                    <ListItemText primary={benefit} />
                  </ListItem>
                ))}
              </List>
            </Paper>
          </Grid>
          <Grid item xs={12} md={6}>
            <Paper elevation={2} sx={{ p: 3, height: '100%' }}>
              <Typography variant="h6" gutterBottom>
                For Public Safety
              </Typography>
              <List>
                {benefits.slice(3).map((benefit, index) => (
                  <ListItem key={index} sx={{ py: 0.5 }}>
                    <ListItemIcon>
                      <SecurityIcon color="primary" />
                    </ListItemIcon>
                    <ListItemText primary={benefit} />
                  </ListItem>
                ))}
              </List>
            </Paper>
          </Grid>
        </Grid>
      </Box>

      {/* Technology Stack */}
      <Box sx={{ mb: 6 }}>
        <Typography variant="h4" component="h2" gutterBottom align="center" sx={{ mb: 4 }}>
          Technology Stack
        </Typography>
        <Grid container spacing={3}>
          <Grid item xs={12} sm={6} md={3}>
            <Card elevation={2}>
              <CardContent sx={{ textAlign: 'center' }}>
                <Typography variant="h6" gutterBottom>
                  Backend
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Python Flask, TensorFlow, BERT, Scikit-learn
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card elevation={2}>
              <CardContent sx={{ textAlign: 'center' }}>
                <Typography variant="h6" gutterBottom>
                  Frontend
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  React.js, Material-UI, Chart.js
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card elevation={2}>
              <CardContent sx={{ textAlign: 'center' }}>
                <Typography variant="h6" gutterBottom>
                  ML Models
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  BERT, CNN, Ensemble Methods
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card elevation={2}>
              <CardContent sx={{ textAlign: 'center' }}>
                <Typography variant="h6" gutterBottom>
                  APIs
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Twitter API, News API, Emergency Services
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Box>

      {/* Call to Action */}
      <Box sx={{ textAlign: 'center', py: 4 }}>
        <Typography variant="h5" gutterBottom>
          Ready to Get Started?
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
          Start analyzing tweets for fake news detection and disaster classification.
        </Typography>
        <Button
          variant="contained"
          size="large"
          onClick={() => navigate('/analyze')}
          sx={{ mr: 2 }}
        >
          Try Analysis Now
        </Button>
        <Button
          variant="outlined"
          size="large"
          onClick={() => navigate('/dashboard')}
        >
          View Dashboard
        </Button>
      </Box>
    </Container>
  );
};

export default Home; 