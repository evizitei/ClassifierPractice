# compute the mean and covariance for each class
load data/data_class4.mat
cls1 = Data{1}
cls2 = Data{2}
cls3 = Data{3}
cls4 = Data{4}

len1 = columns(cls1)
len2 = columns(cls2)
len3 = columns(cls3)
len4 = columns(cls4)

mean1 = sum(cls1, 2) / len1
mean2 = sum(cls2, 2) / len2
mean3 = sum(cls3, 2) / len3
mean4 = sum(cls4, 2) / len4

function cov_mat = two_var_cov (data_arr)
  len = columns(data_arr)
  mn = sum(data_arr, 2) / len
  cv_11 = sum((data_arr(1, :) - mn(1)).^ 2, 2)/ len
  cv_12 = sum((data_arr(1, :) - mn(1)).* (data_arr(2, :) - mn(2)), 2)/ len
  cv_22 = sum((data_arr(2, :) - mn(2)).^ 2, 2)/ len
  cov_mat = [cv_11, cv_12; cv_12, cv_22]
endfunction

cov1 = two_var_cov(cls1)
cov2 = two_var_cov(cls2)
cov3 = two_var_cov(cls3)
cov4 = two_var_cov(cls4)

# Compare the covariance you obtainedwith the one obtained
# with the Matlab built-in function'cov'.
# Did you obtain the same results?
# ---
# NO, because of floating point precision issues.
# roundoff and other factors common to numerical
# computing are preventing us from obtaining the
# precise same answer, but they are very close.


# b) Compute eigenvectors and eigenvalues for each class
function eigencells = compute_eigendecomposition (covar_matrix)
  # solve Av = kv where l is some eigenvalue.  v is the eigenvector
  #    so Av - kv = 0, and Av - kIv = 0 (I is identity)
  #    => (A - kI)v = 0
  #  for v to be non-zero, the determinant of (A - kI) must be 0
  #    so det(A - kI) = 0
  #       and (A - kI) is
  #    [ a-k b
  #      c   d-k ]
  #    and therefore
  #     (a-k) * (d-k) - bc = 0
  #    => ad - ak - dk + k^2 - bc = 0
  #    => k^2 - (a-d)k + (ad - bc) = 0
  a = covar_matrix(1,1)
  b = covar_matrix(1,2)
  c = covar_matrix(2,1)
  d = covar_matrix(2,2)
  #    Can be addressed via quadratic equation:
  #       (-b +/- sqrt(b^2 - 4ac)) / 2a
  qa = 1
  qb = (a+d) * -1
  qc = (a*d) - (b*c)
  qsqrt = sqrt((qb*qb) - (4*qa*qc))
  eigenval_1 = ((qb*-1) + qsqrt) / (2*qa)
  eigenval_2 = ((qb*-1) - qsqrt) / (2*qa)
  #    This is a quadratic equation and therefore has 2 solutions for k
  #    Each vector can be solved by
  #      (A - K[i]*I)v = 0
  characteristic_1 = covar_matrix - (eigenval_1 * eye(2))
  characteristic_2 = covar_matrix - (eigenval_2 * eye(2))
  #    This means that v is in the NULL SPACE of (A - kI)
  #       B = (A - K[i]*I) => Bv = 0
  #    So B11*v1 + B12*v2 = 0
  #       B21*v1 + B22*v2 = 0
  #    This is a homogeneous equation which has infinite solutions,
  #      so just set a value and solve for the other
  #      B11*v1 = -1*B12*v2
  #    v1 = 1
  #     B11 + B12*v2 = 0
  #     B21 + B22*v2 = 0
  #    Rewrite to remove the 0:
  #     B12*v2 = -B11
  #    => v2 = -B11 / B12
  vec_1_2 = (-1*characteristic_1(1,1)) / characteristic_1(1,2)
  vec_2_2 = (-1*characteristic_2(1,1)) / characteristic_2(1,2)
  vec_1 = [1; vec_1_2]
  vec_2 = [1; vec_2_2]
  #    Eigenvectors are all of unit length, so normalize by dividing by their norm
  #     e1 = [ 1 (-B11 / B12)] / ||[ 1 (-B11 / B12)]||
  e_vec_1 = vec_1 / sqrt(dot(vec_1, vec_1))
  e_vec_2 = vec_2 / sqrt(dot(vec_2, vec_2))
  eigencells = { e_vec_1, e_vec_2, eigenval_1, eigenval_2 }
endfunction

eigen_class_1 = compute_eigendecomposition(cov1)
eigen_class_2 = compute_eigendecomposition(cov2)
eigen_class_3 = compute_eigendecomposition(cov3)
eigen_class_4 = compute_eigendecomposition(cov4)

x1_vals = [ cls1(1, :) cls2(1, :) cls3(1, :) cls4(1, :) ]
x2_vals = [ cls1(2, :) cls2(2, :) cls3(2, :) cls4(2, :) ]
v1 = cls1(1, :)
v2 = cls2(1, :)
v3 = cls3(1, :)
v4 = cls4(1, :)
v1(:) = 10
v2(:) = 20
v3(:) = 30
v4(:) = 40
color_vals = [ v1 v2 v3 v4 ]
xs=[mean1(1) mean2(1) mean3(1) mean4(1)]
ys=[mean1(2) mean2(2) mean3(2) mean4(2)]
u= [eigen_class_1{1}(1) eigen_class_2{1}(1) eigen_class_3{1}(1) eigen_class_4{1}(1)]
v= [eigen_class_1{1}(2) eigen_class_2{1}(2) eigen_class_3{1}(2) eigen_class_4{1}(2)]
vecColor = 'r'
headSize = 0.15
vecWidth = 3
# use for size and color
scatter(x1_vals, x2_vals, color_vals, color_vals)
title ("4 class scatter plots with first principal vectors (normalized)");
hold on
k=1;
h=quiver(xs(k),ys(k),u(k),v(k), 0,vecColor);
set (h, "maxheadsize", headSize);
set (h, "linewidth", vecWidth)
k=2;
h=quiver(xs(k),ys(k),u(k),v(k), 0,vecColor);
set (h, "maxheadsize", headSize);
set (h, "linewidth", vecWidth)
k=3;
h=quiver(xs(k),ys(k),u(k),v(k), 0,vecColor);
set (h, "maxheadsize", headSize);
set (h, "linewidth", vecWidth)
k=4;
h=quiver(xs(k),ys(k),u(k),v(k), 0,vecColor);
set (h, "maxheadsize", headSize);
set (h, "linewidth", vecWidth)
hold off
