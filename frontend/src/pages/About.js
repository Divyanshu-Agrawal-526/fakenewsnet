import React from 'react';
import {
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  Box,
  Paper,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
} from '@mui/material';
import {
  School as SchoolIcon,
  Code as CodeIcon,
  Psychology as PsychologyIcon,
  Security as SecurityIcon,
  Speed as SpeedIcon,
  CheckCircle as AccuracyIcon,
} from '@mui/icons-material';

const About = () => {
  const features = [
    {
      title: 'Multimodal Analysis',
      description: 'Combines text and image analysis for comprehensive fake news detection',
      icon: <PsychologyIcon color="primary" />,
    },
    {
      title: 'Real-time Processing',
      description: 'Instant analysis of tweets with immediate results and alerts',
      icon: <SpeedIcon color="primary" />,
    },
    {
      title: 'High Accuracy',
      description: 'Advanced ML models achieving over 85% accuracy in fake news detection',
      icon: <AccuracyIcon color="primary" />,
    },
    {
      title: 'Authority Integration',
      description: 'Direct communication with emergency services for verified disasters',
      icon: <SecurityIcon color="primary" />,
    },
  ];

  const technologyStack = [
    {
      category: 'Backend',
      technologies: ['Python Flask', 'TensorFlow', 'PyTorch', 'BERT', 'Scikit-learn'],
    },
    {
      category: 'Frontend',
      technologies: ['React.js', 'Material-UI', 'Chart.js', 'Axios'],
    },
    {
      category: 'Machine Learning',
      technologies: ['BERT', 'CNN', 'Ensemble Methods', 'NLP', 'Computer Vision'],
    },
    {
      category: 'APIs & Services',
      technologies: ['Twitter API', 'News API', 'Geocoding', 'Weather API'],
    },
  ];

  const methodology = [
    'Text preprocessing and feature extraction',
    'BERT-based fake news classification',
    'Disaster type classification using ensemble methods',
    'Fact-checking through multiple news sources',
    'Location-based authority identification',
    'Multi-channel alert system for emergency services',
  ];

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        About the Project
      </Typography>
      
      <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
        A comprehensive capstone project for fake news detection during natural disasters
      </Typography>

      {/* Project Overview */}
      <Card elevation={3} sx={{ mb: 4 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
            <SchoolIcon sx={{ fontSize: 40, color: 'primary.main', mr: 2 }} />
            <Box>
              <Typography variant="h5" component="h2" gutterBottom>
                Final Year Capstone Project
              </Typography>
              <Typography variant="body1" color="text.secondary">
                Fake News Detection During Natural Disasters with Multimodal Classification
              </Typography>
            </Box>
          </Box>
          
          <Typography variant="body1" paragraph>
            This project addresses the critical challenge of fake news during natural disasters, 
            where misinformation can have severe consequences for public safety and emergency response. 
            The system combines advanced machine learning techniques with real-time fact-checking to 
            provide accurate, reliable information to authorities and the public.
          </Typography>
          
          <Typography variant="body1">
            The project demonstrates the application of modern AI/ML technologies in solving 
            real-world problems, with a focus on social impact and public safety.
          </Typography>
        </CardContent>
      </Card>

      {/* Key Features */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h5" component="h2" gutterBottom>
          Key Features
        </Typography>
        <Grid container spacing={3}>
          {features.map((feature, index) => (
            <Grid item xs={12} sm={6} md={3} key={index}>
              <Card elevation={2} sx={{ height: '100%' }}>
                <CardContent sx={{ textAlign: 'center' }}>
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

      {/* Technology Stack */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h5" component="h2" gutterBottom>
          Technology Stack
        </Typography>
        <Grid container spacing={3}>
          {technologyStack.map((stack, index) => (
            <Grid item xs={12} sm={6} md={3} key={index}>
              <Card elevation={2}>
                <CardContent>
                  <Typography variant="h6" component="h3" gutterBottom>
                    {stack.category}
                  </Typography>
                  <List dense>
                    {stack.technologies.map((tech, techIndex) => (
                      <ListItem key={techIndex} sx={{ py: 0 }}>
                        <ListItemIcon>
                          <CodeIcon color="primary" />
                        </ListItemIcon>
                        <ListItemText primary={tech} />
                      </ListItem>
                    ))}
                  </List>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Box>

      {/* Methodology */}
      <Grid container spacing={4}>
        <Grid item xs={12} md={6}>
          <Paper elevation={2} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Methodology
            </Typography>
            <List>
              {methodology.map((step, index) => (
                <ListItem key={index} sx={{ py: 0.5 }}>
                  <ListItemIcon>
                    <Box
                      sx={{
                        width: 24,
                        height: 24,
                        borderRadius: '50%',
                        bgcolor: 'primary.main',
                        color: 'white',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: 12,
                        fontWeight: 'bold',
                      }}
                    >
                      {index + 1}
                    </Box>
                  </ListItemIcon>
                  <ListItemText primary={step} />
                </ListItem>
              ))}
            </List>
          </Paper>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Paper elevation={2} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Project Objectives
            </Typography>
            <List>
              <ListItem sx={{ py: 0.5 }}>
                <ListItemText 
                  primary="Detect fake news in disaster-related tweets"
                  secondary="Using advanced NLP and ML techniques"
                />
              </ListItem>
              <ListItem sx={{ py: 0.5 }}>
                <ListItemText 
                  primary="Classify real disasters by type"
                  secondary="Wildfire, flood, hurricane, earthquake"
                />
              </ListItem>
              <ListItem sx={{ py: 0.5 }}>
                <ListItemText 
                  primary="Verify information authenticity"
                  secondary="Through fact-checking and multiple sources"
                />
              </ListItem>
              <ListItem sx={{ py: 0.5 }}>
                <ListItemText 
                  primary="Enable authority contact"
                  secondary="Direct communication with emergency services"
                />
              </ListItem>
            </List>
          </Paper>
        </Grid>
      </Grid>

      {/* Technical Details */}
      <Card elevation={3} sx={{ mt: 4 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Technical Implementation Details
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle1" gutterBottom>
                Machine Learning Models
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                • BERT-based text classification for fake news detection<br/>
                • Ensemble methods combining multiple ML algorithms<br/>
                • CNN for image analysis and disaster detection<br/>
                • TF-IDF and feature engineering for text processing
              </Typography>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle1" gutterBottom>
                System Architecture
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                • Flask REST API backend with modular design<br/>
                • React.js frontend with Material-UI components<br/>
                • SQLite database for data persistence<br/>
                • Real-time processing and alert system
              </Typography>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Academic Information */}
      <Card elevation={2} sx={{ mt: 4 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Academic Information
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            This project was developed as a final year capstone project, demonstrating the application 
            of computer science and artificial intelligence concepts to solve real-world problems. 
            The project showcases skills in machine learning, web development, API integration, 
            and system design.
          </Typography>
          
          <Typography variant="body2" color="text.secondary">
            <strong>Note:</strong> This is an academic project and is not intended for commercial use. 
            All APIs and services used are for demonstration purposes only.
          </Typography>
        </CardContent>
      </Card>
    </Container>
  );
};

export default About; 