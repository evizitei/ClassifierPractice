# defined as sqrt((v1 - v2)^T * SIGMA^-1 * (v1 - v2))
# if v2 could be the mean of the distribution, or a second
# sample under the same distribution.
# Interesting note, euclidean distance is sqrt((v1 - v2)^T * (v1 - v2))
# which would be mahalanobis with an identity matrix covariance,
# so euclidean distance is a special case of mahalanobis distance.
function mahal_dist = compute_mahalanobis_distance(v1, v2, covar_matrix)
    vector_delta = (v1 - v2)
    covar_inv = inv(covar_matrix)
    mahal_dist = sqrt((vector_delta') * (covar_inv * vector_delta))
endfunction

# generic form of discriminant function is:
# g_i(x) = (-1/2)mahal_dist(x, mean, covar) - (d/2)*ln(2*pi) - (1/2)*ln(det(covar)) + ln(Prior)
function discriminant_val = discriminant_function(observation, mean_vec, covar_matrix, prior)
    dimensionality = size(observation)(1)
    mahal_d = compute_mahalanobis_distance(observation, mean_vec, covar_matrix)
    t1 = (-1/2)*mahal_d
    t2 = (dimensionality/2)*log(2*pi)
    t3 = (1/2)*log(det(covar_matrix))
    t4 = log(prior)
    discriminant_val = t1 - t2 - t3 + t4
endfunction

function mean_vector = compute_mean(data_arr)
    sample_count = columns(data_arr)
    mean_vector = sum(data_arr, 2) / sample_count
endfunction

function cov_mat = compute_cov (data_arr)
  len = columns(data_arr)
  dim = rows(data_arr)
  mn = sum(data_arr, 2) / len
  cov_mat = zeros(dim, dim)
  for i = 1:dim
    for j = 1:dim
      cov_mat(i,j) = sum((data_arr(i, :) - mn(i)).* (data_arr(j, :) - mn(j)), 2)/ (len - 1)
    endfor
  endfor
endfunction

class_1_data = [
    -5.01, -8.12, -3.68;
    -5.43, -3.48, -3.54;
    1.08, -5.52, 1.66;
    0.86, -3.78, -4.11;
    -2.67, 0.63, 7.39;
    4.94, 3.29, 2.08;
    -2.51, 2.09, -2.59;
    -2.25, -2.13, -6.94;
    5.56, 2.86, -2.26;
    1.03, -3.33, 4.33;
]'

class_2_data = [
    -0.91, -0.18, -0.05;
    1.3, -2.06, -3.53;
    -7.75, -4.54, -0.95;
    -5.47, 0.5, 3.92;
    6.14, 5.72, -4.85;
    3.6, 1.26, 4.36;
    5.37, -4.63, -3.65;
    7.18, 1.46, -6.66;
    -7.39, 1.17, 6.3;
    -7.5, -6.32, -0.31
]'

class_3_data = [
    5.35, 2.26, 8.13;
    5.12, 3.22, -2.66;
    -1.34, -5.31, -9.87;
    4.48, 3.42, 5.19;
    7.11, 2.39, 9.21;
    7.17, 4.33, -0.98;
    5.75, 3.97, 6.65;
    0.77, 0.27, 2.41;
    0.9, -0.43, -8.71;
    3.52, -0.36, 6.43
]'

class_1_prior = 0.6
class_2_prior = 0.2
class_3_prior = 0.2
class_1_mean = compute_mean(class_1_data)
class_2_mean = compute_mean(class_2_data)
class_3_mean = compute_mean(class_3_data)
cv_1 = compute_cov(class_1_data)
cv_2 = compute_cov(class_2_data)
cv_3 = compute_cov(class_3_data)

point1 = [ 1; 3; 2]
point2 = [ 4; 6; 1]
point3 = [ 7;-1; 0]
point4 = [-2; 6; 5]

g_1_1 = discriminant_function(point1, class_1_mean, cv_1, class_1_prior)
g_2_1 = discriminant_function(point1, class_2_mean, cv_2, class_2_prior)
g_3_1 = discriminant_function(point1, class_3_mean, cv_3, class_3_prior)
g_1_2 = discriminant_function(point2, class_1_mean, cv_1, class_1_prior)
g_2_2 = discriminant_function(point2, class_2_mean, cv_2, class_2_prior)
g_3_2 = discriminant_function(point2, class_3_mean, cv_3, class_3_prior)
g_1_3 = discriminant_function(point3, class_1_mean, cv_1, class_1_prior)
g_2_3 = discriminant_function(point3, class_2_mean, cv_2, class_2_prior)
g_3_3 = discriminant_function(point3, class_3_mean, cv_3, class_3_prior)
g_1_4 = discriminant_function(point4, class_1_mean, cv_1, class_1_prior)
g_2_4 = discriminant_function(point4, class_2_mean, cv_2, class_2_prior)
g_3_4 = discriminant_function(point4, class_3_mean, cv_3, class_3_prior)

results_1 = [g_1_1, g_2_1, g_3_1]
results_2 = [g_1_2, g_2_2, g_3_2]
results_3 = [g_1_3, g_2_3, g_3_3]
results_4 = [g_1_4, g_2_4, g_3_4]

test_point = class_2_mean
g_1_t = discriminant_function(test_point, class_1_mean, cv_1, class_1_prior)
g_2_t = discriminant_function(test_point, class_2_mean, cv_2, class_2_prior)
g_3_t = discriminant_function(test_point, class_3_mean, cv_3, class_3_prior)


function eigencells = compute_eigendecomposition (covar_matrix)
  a = covar_matrix(1,1)
  b = covar_matrix(1,2)
  c = covar_matrix(2,1)
  d = covar_matrix(2,2)
  qa = 1
  qb = (a+d) * -1
  qc = (a*d) - (b*c)
  qsqrt = sqrt((qb*qb) - (4*qa*qc))
  eigenval_1 = ((qb*-1) + qsqrt) / (2*qa)
  eigenval_2 = ((qb*-1) - qsqrt) / (2*qa)
  characteristic_1 = covar_matrix - (eigenval_1 * eye(2))
  characteristic_2 = covar_matrix - (eigenval_2 * eye(2))
  vec_1_2 = (-1*characteristic_1(1,1)) / characteristic_1(1,2)
  vec_2_2 = (-1*characteristic_2(1,1)) / characteristic_2(1,2)
  vec_1 = [1; vec_1_2]
  vec_2 = [1; vec_2_2]
  e_vec_1 = vec_1 / sqrt(dot(vec_1, vec_1))
  e_vec_2 = vec_2 / sqrt(dot(vec_2, vec_2))
  eigencells = { e_vec_1, e_vec_2, eigenval_1, eigenval_2 }
endfunction


# this first makes a set of samples from a standard normal distribution
# (centered at 0, covariance of Identity matrix)
# to make this work for the specified distribution
# we need to de-whitening-transform each point to match the
# provided covariance matrix and then shift it to the right centrality
# by adding the mean.
#
# The whitening transform is defined as:
# OrthonormalEigenvectors * diagonal_eigenvalues^-1/2
# intuitively, this maps a given point onto a spherical normal distribution.
# Therefore the inverse of this matrix would transform points from a spherical
# normal distribution to a specified covariance matrix.
#
# Acutally, the covariance matrix in this exercise is diagonal,
# so the eigenvectors are just the standard basis vectos
# and the eigenvalues are the diagonal values.
function output_dataset = generate_normal_sample (mean_vec, covar_mat, smpl_count)
  dim = rows(mean_vec)
  init_sample = randn(dim, smpl_count)
  ev1 = [1; 0]
  ev2 = [0; 1]
  diag_evals = [(1 / sqrt(covar_mat(1,1))), 0; 0, (1 / sqrt(covar_mat(2,2)))]
  evecs = [ev1, ev2]
  whitening_transform = evecs * diag_evals
  inv_whit_transform = inv(whitening_transform)
  skewed_data = inv_whit_transform * init_sample
  shifted_sample = skewed_data + mean_vec
  output_dataset = shifted_sample
endfunction

# 3D plotting requires having some value for "height", the probability of the
# given point seems appropriate.
# P(x| mu, sigma) =
#  1/{[(2*pi)^(d/2)]*sqrt(det(sigma))} * e^[(-1/2)*(x-mu)^t*inv(sigma)*(x-mu)]

function pw1 = class_1_likelihood (x, y)
  mean_vec = [8; 2];
  covar_mat = [4.1, 0; 0, 2.8];
  m_delt = ([x; y] - mean_vec);
  covar_det = det(covar_mat);
  covar_inv = inv(covar_mat);
  t1 = (1 / ((2*pi)* sqrt(covar_det)));
  t2 = exp((-1/2)*(m_delt' * covar_inv * m_delt));
  pw1 = t1 * t2;
endfunction

function pw2 = class_2_likelihood (x, y)
  mean_vec = [2; 8];
  covar_mat = [4.1, 0; 0, 2.8];
  m_delt = ([x; y] - mean_vec);
  covar_det = det(covar_mat);
  covar_inv = inv(covar_mat);
  t1 = (1 / ((2*pi)* sqrt(covar_det)));
  t2 = exp((-1/2)*(m_delt' * covar_inv * m_delt));
  pw2 = t1 * t2;
endfunction

grid_side_count = 100;
point_range = linspace (-5, 12, grid_side_count);
[X, Y] = meshgrid (point_range, point_range);
#Z = arrayfun(class_1_likelihood, X, Y);
#surf (X, Y, Z);
# TODO: figure out how to vectorize this
Z1 = zeros(grid_side_count, grid_side_count);
for i = 1:grid_side_count
  for j = 1:grid_side_count
    Z1(i,j) = class_1_likelihood(X(i,j), Y(i,j));
  endfor
endfor

Z2 = zeros(grid_side_count, grid_side_count);
for i = 1:grid_side_count
  for j = 1:grid_side_count
    Z2(i,j) = class_2_likelihood(X(i,j), Y(i,j));
  endfor
endfor
max_val_surface = max(Z1, Z2);
mesh_color_surface = (Z1 > Z2) .* linspace(0.5,0.6,grid_side_count)


c1_mean_vec = [8; 2];
c2_mean_vec = [2; 8];
both_covar_mat = [4.1, 0; 0, 2.8];

data_sample_1 = generate_normal_sample(c1_mean_vec, both_covar_mat, 1000);
c1_x = data_sample_1(1, :);
c1_y = data_sample_1(2, :);


data_sample_2 = generate_normal_sample(c2_mean_vec, both_covar_mat, 1000);
c2_x = data_sample_2(1, :);
c2_y = data_sample_2(2, :);

y_offset = 0.05;
c1_z = zeros(1, columns(c1_y)) - y_offset;
c2_z = zeros(1, columns(c2_y)) - y_offset;

mesh(X, Y, max_val_surface, mesh_color_surface);
hidden off
hold on
point_size = 2;
scatter3(c1_x, c1_y, c1_z, point_size, 'x');
scatter3(c2_x, c2_y, c2_z, point_size, 'x');
hold off

# for question 2 (b), the covariance matrices
# are the same between classes.  Per Case 2 (2.6.2)
# this means the discriminant function for each class can
# be written as:
#
# g_i(x) = -(1/2)*(x-mu_i)^T*(sigma_inverse)*(x-mu_i) + ln(P(class_i))
# working through the derivation to page 40, we get the equation for
# the decision boundary is w^t(x - x_0) = 0
#  - to plot this, we need x2 as a function of x1, so we expand and solve for x_2
#
# x_2 = (1/w_2)(w^t*x_0)-(w_1/w_2)*x_1
#
# these values have big expansions, but much can be precomputed.
#
# x_0= (1/2)*(class_1_mean + class_2_mean) -
#         [(ln(P(w1)/P(w2)))/((c1_mean - c2_mean)^t*sig_inv*(c1_mean-c2_mean)] * (c1_mean - c2_mean)
#
# w = sig_inv * (c1_mean - c2_mean)

c1_prior = 0.8
c2_prior = 0.2
prior_ratio = log(c1_prior / c2_prior)
sig_inv = inv(both_covar_mat)
mean_delta = (c1_mean_vec - c2_mean_vec)

x_0__t1 = (1/2)*(c1_mean_vec + c2_mean_vec)
x_0__t2 = (prior_ratio/(mean_delta' * sig_inv * mean_delta))*mean_delta
x_0 = x_0__t1 - x_0__t2
w = sig_inv * mean_delta
w_ratio = (w(1)/w(2))
y_bias = (1/w(2))*(w'*x_0)

boundary_y_from_x = @(x) y_bias - (w_ratio)*x

boundary_x_1_vals = linspace (-10, 15, grid_side_count);
boundary_x_2_vals = boundary_y_from_x(boundary_x_1_vals);
boundary_z_vals = zeros(1, columns(boundary_x_2_vals)) - y_offset;
hold on
plot3(boundary_x_1_vals, boundary_x_2_vals, boundary_z_vals, '-or');
hold off


c1_posterior = Z1 .* c1_prior;
c2_posterior = Z2 .* c2_prior;
posterior_surface = max(c1_posterior, c2_posterior);
posterior_color_surface = (c1_posterior > c2_posterior) .* linspace(0.5,0.6,grid_side_count);
mesh(X, Y, posterior_surface, posterior_color_surface);
hidden off
hold on
point_size = 2;
scatter3(c1_x, c1_y, c1_z, point_size, 'x');
scatter3(c2_x, c2_y, c2_z, point_size, 'x');
hold off