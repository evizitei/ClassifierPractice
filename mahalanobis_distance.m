function mahal_dist = compute_mahalanobis_distance(v1, v2, covar_matrix)
    # defined as sqrt((v1 - v2)^T * SIGMA^-1 * (v1 - v2))
    # if v2 could be the mean of the distribution, or a second
    # sample under the same distribution.
    # Interesting note, euclidean distance is sqrt((v1 - v2)^T * (v1 - v2))
    # which would be mahalanobis with an identity matrix covariance,
    # so euclidean distance is a special case of mahalanobis distance.
    vector_delta = (v1 - v2)
    covar_inv = inv(covar_matrix)
    mahal_dist = sqrt((vector_delta') * (covar_inv * vector_delta))
endfunction

function discriminant_val = discriminant_function(observation, mean_vec, covar_matrix, prior)
    # generic form of discriminant function is:
    # g_i(x) = (-1/2)mahal_dist(x, mean, covar) - (d/2)*ln(2*pi) - (1/2)*ln(det(covar)) + ln(Prior)
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
