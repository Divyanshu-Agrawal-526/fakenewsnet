import React from 'react';
import {
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  Box,
  Paper,
  CircularProgress,
} from '@mui/material';
import {
  TrendingUp as TrendingIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Category as CategoryIcon,
} from '@mui/icons-material';
import { useQuery } from 'react-query';
import axios from 'axios';
import { Line, Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

const Dashboard = () => {
  const { data: stats, isLoading, error } = useQuery('statistics', async () => {
    const response = await axios.get('/api/statistics');
    return response.data;
  });

  const mockData = {
    total_analyses: 1250,
    accuracy: 0.87,
    disaster_distribution: {
      wildfire: 45,
      flood: 30,
      hurricane: 15,
      earthquake: 10,
    },
    system_uptime: new Date().toISOString(),
    models_status: {
      fake_news_detector: true,
      disaster_classifier: true,
      fact_checker: true,
    },
  };

  const data = stats || mockData;

  const chartData = {
    labels: ['Wildfire', 'Flood', 'Hurricane', 'Earthquake'],
    datasets: [
      {
        data: [
          data.disaster_distribution?.wildfire || 0,
          data.disaster_distribution?.flood || 0,
          data.disaster_distribution?.hurricane || 0,
          data.disaster_distribution?.earthquake || 0,
        ],
        backgroundColor: [
          '#ff6b6b',
          '#4ecdc4',
          '#45b7d1',
          '#96ceb4',
        ],
        borderWidth: 2,
        borderColor: '#fff',
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'bottom',
      },
      title: {
        display: true,
        text: 'Disaster Type Distribution',
      },
    },
  };

  const StatCard = ({ title, value, icon, color, subtitle }) => (
    <Card elevation={2}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <Box sx={{ color, mr: 2 }}>
            {icon}
          </Box>
          <Typography variant="h6" component="div">
            {title}
          </Typography>
        </Box>
        <Typography variant="h4" component="div" sx={{ fontWeight: 'bold', color }}>
          {value}
        </Typography>
        {subtitle && (
          <Typography variant="body2" color="text.secondary">
            {subtitle}
          </Typography>
        )}
      </CardContent>
    </Card>
  );

  if (isLoading) {
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 400 }}>
          <CircularProgress />
        </Box>
      </Container>
    );
  }

  if (error) {
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Typography variant="h6" color="error" align="center">
          Error loading dashboard data
        </Typography>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Dashboard
      </Typography>
      
      <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
        System statistics and performance metrics
      </Typography>

      {/* Statistics Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Total Analyses"
            value={data.total_analyses?.toLocaleString() || '0'}
            icon={<TrendingIcon />}
            color="primary.main"
            subtitle="Tweets analyzed"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Accuracy"
            value={`${((data.accuracy || 0) * 100).toFixed(1)}%`}
            icon={<CheckCircleIcon />}
            color="success.main"
            subtitle="Model accuracy"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="System Status"
            value="Online"
            icon={<CheckCircleIcon />}
            color="success.main"
            subtitle="All systems operational"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Models Loaded"
            value={Object.values(data.models_status || {}).filter(Boolean).length}
            icon={<CategoryIcon />}
            color="info.main"
            subtitle="Active ML models"
          />
        </Grid>
      </Grid>

      {/* Charts */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper elevation={2} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Disaster Type Distribution
            </Typography>
            <Box sx={{ height: 300 }}>
              <Doughnut data={chartData} options={chartOptions} />
            </Box>
          </Paper>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Paper elevation={2} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Model Status
            </Typography>
            <Box sx={{ mt: 2 }}>
              {Object.entries(data.models_status || {}).map(([model, status]) => (
                <Box key={model} sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <CheckCircleIcon 
                    color={status ? 'success' : 'error'} 
                    sx={{ mr: 1 }} 
                  />
                  <Typography variant="body2">
                    {model.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </Typography>
                </Box>
              ))}
            </Box>
          </Paper>
        </Grid>
      </Grid>

      {/* System Information */}
      <Grid container spacing={3} sx={{ mt: 2 }}>
        <Grid item xs={12}>
          <Paper elevation={2} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              System Information
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <Typography variant="body2" color="text.secondary">
                  <strong>System Uptime:</strong> {new Date(data.system_uptime).toLocaleString()}
                </Typography>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography variant="body2" color="text.secondary">
                  <strong>Last Updated:</strong> {new Date().toLocaleString()}
                </Typography>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default Dashboard; 