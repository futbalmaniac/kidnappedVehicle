/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <limits>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::uniform_int_distribution;
using std::uniform_real_distribution;
using std::numeric_limits;

static std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  if (is_initialized) {
    return;
  }

  num_particles = 100;  // TODO: Set the number of particles
  normal_distribution<double> dist_x_init(x, std[0]);
  normal_distribution<double> dist_y_init(y, std[1]);
  normal_distribution<double> dist_theta_init(theta, std[2]);

  for (int i=0;i<num_particles;i++){
    Particle p;
    p.id=i;
    p.x=dist_x_init(gen);
    p.y=dist_y_init(gen);
    p.theta=dist_theta_init(gen);
    p.weight=1.0;
    particles.push_back(p);

  }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
   normal_distribution<double> dist_x(0, std_pos[0]);
   normal_distribution<double> dist_y(0, std_pos[1]);
   normal_distribution<double> dist_theta(0, std_pos[2]);
  for (int i=0;i<num_particles;i++){
     
    if (fabs(yaw_rate)<0.00001){
      particles[i].x += velocity* delta_t *cos(particles[i].theta);
      particles[i].y += velocity* delta_t *sin(particles[i].theta); 
    }
      else{
        particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      	particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      	particles[i].theta += yaw_rate * delta_t;
      }
        
    particles[i].x +=dist_x(gen);
    particles[i].y +=dist_y(gen);
    particles[i].theta +=dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (unsigned int i = 0; i < observations.size(); i++) {
    double min_dist = std::numeric_limits<double>::max();
    int map_id = -1;
    
    for (unsigned int j = 0; j < predicted.size(); j++) {
      double xDistance = observations[i].x - predicted[j].x;
      double yDistance = observations[i].y - predicted[j].y;
      double cur_dist = xDistance * xDistance + yDistance * yDistance;
      if (cur_dist < min_dist) {
        min_dist = cur_dist;
        map_id = predicted[j].id;
      }
    }
    observations[i].id = map_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
	for (int i = 0; i < num_particles; i++) {

    double particle_x = particles[i].x;
    double particle_y = particles[i].y;
    double particle_theta = particles[i].theta;
    double sensor_range_2 = sensor_range * sensor_range;

    vector<LandmarkObs> predictions;
      
    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {

      float landmark_x = map_landmarks.landmark_list[j].x_f;
      float landmark_y = map_landmarks.landmark_list[j].y_f;
      int landmark_id = map_landmarks.landmark_list[j].id_i;
      double dX = landmark_x-particle_x;
      double dY = landmark_y-particle_y;
      
      
      if ( dX*dX + dY*dY <= sensor_range_2 ) {
        predictions.push_back(LandmarkObs{ landmark_id, landmark_x, landmark_y });
      }
    }

    
    vector<LandmarkObs> transformed_os;
    for (unsigned int j = 0; j < observations.size(); j++) {
      double transformed_x = cos(particle_theta)*observations[j].x - sin(particle_theta)*observations[j].y + particle_x;
      double transformed_y = sin(particle_theta)*observations[j].x + cos(particle_theta)*observations[j].y + particle_y;
      transformed_os.push_back(LandmarkObs{ observations[j].id, transformed_x, transformed_y });
    }
      
    dataAssociation(predictions, transformed_os);

    particles[i].weight = 1.0;

    for (unsigned int j = 0; j < transformed_os.size(); j++) {
      double observation_x, observation_y, prediction_x, prediction_y;
      observation_x = transformed_os[j].x;
      observation_y = transformed_os[j].y;

      int associated_prediction = transformed_os[j].id;
      for (unsigned int k = 0; k < predictions.size(); k++) {
        if (predictions[k].id == associated_prediction) {
          prediction_x = predictions[k].x;
          prediction_y = predictions[k].y;
        }
      }
      
      double std_x = std_landmark[0];
      double std_y = std_landmark[1];
      double dX = observation_x - prediction_x;
      double dY = observation_y - prediction_y;
      double obs_w = ( 1/(2*M_PI*std_x*std_y)) * exp( -( dX*dX/(2*std_x*std_x) + (dY*dY/(2*std_y*std_y)) ) );

      if (obs_w == 0) {
        particles[i].weight *= 0.00001;
      } else {
        particles[i].weight *= obs_w;
      }
    }
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  vector<Particle> new_particles;

  vector<double> weights;
  for (int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
  }

  uniform_int_distribution<int> uniintdist(0, num_particles-1);
  auto index = uniintdist(gen);

  double max_weight =*max_element(weights.begin(), weights.end());

  uniform_real_distribution<double> unirealdist(0.0, max_weight);

  double beta = 0.0;

  for (int i = 0; i < num_particles; i++) {
    beta += unirealdist(gen) * 2.0;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }

  particles = new_particles;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}