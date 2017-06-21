// Author: Ananya Jana
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <math.h>
#define TRUE            1
#define FALSE           0
#define EPSILON         0.0000000000000000000001
#define EPSILON2        0.01
#define NO_OF_CEP		12
#define SAMPLE_IN_FRAME 320
#define SILENCE_ENERGY	481744.00
#define CODEBOOK_SIZE   8
#define N1              5                   // number of states
#define M               8
#define T               200

char* file_name1 = "two_s.txt";
char* file_name2 = "dc_corrected.txt";		// contains the dc corrected data 
char* file_name3 = "normalized_data.txt";	// contains the normalized data 
char* file_name4 = "speech.txt";            // contains the speech data
//char* file_name5 = "LPC_coeffs.txt"
char* file_name6 = "cepstral_coeffs.txt";
char* file_name7 = "train_vector.txt";
char* file_name8 = "code_book.txt";
char* file_name9 = "deviation.txt";
char* file_name10 = "temp_positive.txt";
char* file_name11 = "temp_negative.txt";
char* file_name12 = "avg_distortion.txt";
char* file_name13 = "lbg_log.txt";
char* file_name14 = "hmm_log.txt";
char* file_name15 = "model_set.txt";

// initial processing: chief components                                                    
void calculate_no_samples2();
void silence_removal();
void normalize();
void DCshift();
double* values;
int no_samples;								// contains the no. of samples in the sound data
long int sound_samples;						// conatins the no. of samples after silence removal
int start, m1, m2, max_index;				// m1 marks the beginning of speech and m2 the ending of it.max_index stores the value of the indes\x where value of a sample is maximum
											//start keeps the index from which current frame is starts  

// yes no detection: chief components
void yes_no_detection();
double* data;


// calculation of cepstral co-efficients: chief components
void hamming_window_values();
void apply_ham_win();
void calculate_ci();
void calculate_ri();
void get_frame();
double ri_val[NO_OF_CEP];                   // auto correlation array
double ham_win[SAMPLE_IN_FRAME];			// contains the hamming window values
double current_frame[SAMPLE_IN_FRAME];


// K means algorithm: chief components
double** centroid;                          // this 2D array contains the centroid of the clusters
double prev_distortion;                       // this contains the value of previous distortion
double new_distortion;
double** data_set;							// this 2D array contains the test data set               
double** initial_codebook;					// this 2D array contains the initial codebook
double* min_distance;
int* data_set_index;
int* cluster_length;
int weights[NO_OF_CEP] = {1, 2, 4, 8, 16, 16, 25, 36, 49, 64, 81, 100};
void kmeans();
void update_centroid();
void find_avg_distortion();

//LBG
void LBG();
void initialize_LBG();                     // calculates the centroid of the training vectors which is the 1st entry in codebook
void split_codebook();
double centroid1[NO_OF_CEP];
int N;										// no. of test vector
int K = 1;										// size of codebook i.e. no. of vectors in codebook
FILE* fp_dev;
FILE* fp_dist;
FILE* fp_log;

//HMM
void HMM();
void get_N();
void get_index();
void calculate_index();
void solution_prob1();
void solution_prob2();
void solution_prob3();
void forward_procedure();
void backward_procedure();
double alpha[T][N1];
double beta[T][N1];
double delta[T][N1];
int pi[N1] = {1, 0, 0, 0, 0};
double xi[T][N1][N1];
double xi_sum[N1][N1];
double gamma[T][N1];
double gamma_sum[N1];
double gamma_sum1[M][N1];
int psi[T][N1];
double B[N1][M];
//double A[N1][N1] = {{0, 1, 0, 0, 0}, {0, 1, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}};
//double A[N1][N1] = {{0, 0, 1, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}};
double A[N1][N1] = {{0.8, 0.2, 0, 0, 0}, {0, .8, 0.2, 0, 0}, {0, 0, 0.8, 0.2, 0}, {0, 0, 0, 0.8, 0.2}, {0, 0, 0, 0, 1}};
//double A[N1][N1] =  {{0.2, 0.8, 0, 0, 0}, {0, .2, 0.8, 0, 0}, {0, 0, 0.2, 0.8, 0}, {0, 0, 0, 0.2, 0.8}, {0, 0, 0, 0, 1}};
//double A[N1][N1] = {{0, 1, 0, 0, 0}, {0, 1, 0, 0, 0}, {0, 1, 0, 0, 0}, {0, 1, 0, 0, 0}, {0, 1, 0, 0, 0}};
double product[T];
int O[T];
double P;
double Pstar;
int Qstar[T];
FILE* fp_hmm;
FILE* fp_model;
//********************************************************************************************************************

int main()
{     
    calculate_no_samples2(); 
    
    //printf("\nThe no. of samples in the given sound data is: %d\n", no_samples);
    
    DCshift();
    normalize();
    silence_removal();
    //yes_no_detection();
    hamming_window_values();
    start = 0;
    calculate_ci();
    HMM();
	free(data_set);
	free(initial_codebook);
	free(centroid);
	free(min_distance);
	free(data_set_index);
	delete[]values;
	delete[]data;
	printf("Finished.:)\n");
	getchar();
}

void calculate_no_samples2()
{
    int i, count = 0;						// count keeps track of the number of lines in the file which is equal to the number of samples
    FILE* fp = fopen(file_name1, "r");
    
    if(!fp){
        printf("File %s cannot be opened.", file_name1);
        exit(EXIT_FAILURE);
    }
    
    while(!feof(fp)){
        fscanf(fp, "%d", &i);
        ++count;
    }
    //printf("count = %d\n", count);
    fclose(fp);
    no_samples = (count - 1);
}

void DCshift()
{
    int i;
    double j = 0, c;
   	double avg = 0;
   	values = new double[no_samples];
	FILE* fp1;
	FILE* fp2;

	fp1 = fopen(file_name1, "r");			// this text file contains the values of samples without the header portion
    
    if(!fp1){
        printf("File %s cannot be opened.", file_name1);
        exit(EXIT_FAILURE);
    }
        
    for(i = 0; i < no_samples; ++i){ 
	      fscanf(fp1, "%lf", &c);           // in each iteration of the loop the c contains the value of the next sample read from the file
	      values[i] = c;
          j = j + c;                        // j contains the summation of values of the samples
    }
    //printf("i = %d, c = %d, j = %ld\n",i, c, j); 
    
    avg = j / (double)(no_samples);         // avg is the value of the DC shift
    //printf("\nthe average is %lf\n", avg);
    fp2 = fopen(file_name2, "w");   
    
    if(!fp2){
        printf("File %s cannot be opened.", file_name2);
        exit(EXIT_FAILURE);
    }
											// this text file contains the values of samples without DC shift
    for(i = 0; i < no_samples; ++i){
	      values[i] = values[i] - avg;
	      fprintf(fp2, "%lf\n", values[i]);
    }
    fclose(fp1);
    fclose(fp2);
}

void normalize()
{
    int i;
    double c, c_abs, max = 0.0, c_norm;
    FILE* fp = fopen(file_name2, "r");
    FILE* fp1 = fopen(file_name3, "w");
    
    if((!fp) || (!fp1)){
        printf("File cannot be opened.");
        exit(EXIT_FAILURE);
        }
    
    for(i = 0; i < no_samples; ++i){
          fscanf(fp, "%lf", &c);
          c_abs = fabs(c);                  // c_abs contains the absolute value of sample
          if(c_abs > max){
                   max = c_abs;             // max contains the maximum of all the absolute values of samples
                   max_index = i;
          }
    }
    //printf("\nthe index at which maximum value occurs is: %d\n",max_index); 
    //printf("\nMaximum absolute value of sample: %lf\n", max);
    
    fseek(fp, 0, SEEK_SET);
    for(i = 0; i < no_samples; ++i){
          fscanf(fp, "%lf", &c);
          c_norm = (c*10000)/max;
          fprintf(fp1, "%lf\n", c_norm);
    }
    
    fclose(fp);
    fclose(fp1);
}

void silence_removal()
{
    int i, j;  
    double avg = 0.0;                         
    double c;
    double avg_energy;
    double sample_energy;
    double total_energy = 0.0;
    double sample_energy_avg;
    
    FILE* fp = fopen(file_name3, "r");
    FILE* fp1 = fopen(file_name4, "w");
    
    
    if(!fp1){
        printf("File %s cannot be opened.", file_name4);
        exit(EXIT_FAILURE);
        }
        
    if(!fp){
        printf("File %s cannot be opened.", file_name3);
        exit(EXIT_FAILURE);
    }
    
    for(i = 0; i < no_samples; ++i){
        fscanf(fp, "%lf", &c);
        values[i] = c;
        total_energy +=(c * c);
        //printf("%lf\n", total_energy);
    }
    avg_energy = total_energy / (long double)no_samples;
    //printf("\naverage energy of the samples is: %Lf\n", avg_energy);
    
    for(i = 0; i < no_samples; i += 50){
          sample_energy = 0.0;
          for( j = 0; j < 100; ++j)
               sample_energy += (values[i + j] * values[i + j]);
          sample_energy_avg = sample_energy / 100;
          if(sample_energy_avg > avg_energy * 0.1){
               for(j = 0; j < 100; ++j)
                     if((values[i + j] * values[i + j]) > avg_energy * 0.1){
                          m1 = i + j;
                          //printf("\nthe first marker is placed at: %d\n", m1);
                          break;
                     }
          break;
          }
    }  
    
    for(i = no_samples - 1; i >= 0; i -= 50){
          sample_energy = 0.0;
          for( j = 0; j < 100; ++j)
               sample_energy += (values[i - j] * values[i - j]);
          sample_energy_avg = sample_energy / 100;
          if(sample_energy_avg > avg_energy * 0.1){
               for(j = 0; j < 100; ++j)
                     if((values[i - j] * values[i - j]) > avg_energy * 0.1){
                          m2 = i - j;
                          //printf("\nthe second marker is placed at: %d\n", m2);
                          break;
                     }
          break;
          }
    } 
	sound_samples = m2 - m1 + 1;
	j = 0;
	data = new double[sound_samples];
    
    for(i = m1; i <= m2; ++i){
          fprintf(fp1, "%lf\n", values[i]);
		  data[j] = values[i];
		  ++j;
	}
                     
    fclose(fp);
    fclose(fp1);
}

void hamming_window_values()
{
     int i;
     
     //printf("the hamming window values are: ");
     for(i = 0; i < SAMPLE_IN_FRAME; ++i){
           ham_win[i] = 0.54 - 0.46 * cos((double)(2 * 22 * i) / (double)(7 * 319));
           //printf("%d th value %lf\n", i, ham_win[i]);
     }
}

void calculate_ci()
{
	 int index;
     int i, j, m, k;
     double ei_val[NO_OF_CEP + 1];
     double ki[NO_OF_CEP + 1];
     double ci_val[NO_OF_CEP + 1];
     double alpha_val[NO_OF_CEP + 1][NO_OF_CEP + 1];
     double ai_val[NO_OF_CEP + 1];
     double sum_term = 0;;
     
     FILE* fp = fopen(file_name6, "w");
	 max_index = max_index - m1;
     //start = max_index + 100;
     start = 0;
     //for(index = 0; index < 60; ++index){
     for(index = 0; index < no_samples; ++index){
		 if((start + SAMPLE_IN_FRAME) < sound_samples){
			for(i = 0; i < SAMPLE_IN_FRAME; ++i){
				current_frame[i] = data[i + start];
				//printf("%lf ", current_frame[i]);
			}
     
			start += 80;
			//get_frame();
			apply_ham_win();
			calculate_ri();
     
			if(ri_val[0] <= SILENCE_ENERGY){
				printf("no need for further processing.\n");
				exit(0);
			}
     
			ei_val[0] = ri_val[0];
			ki[1] = ri_val[1] / ri_val[0];
			alpha_val[1][1] = ki[1];
			ei_val[1] = (1 - (ki[1] * ki[1]))* ei_val[0];
     
			for(i = 2; i < NO_OF_CEP + 1; ++i){
				sum_term = 0.0;
				for(j = 1; j <= i - 1; ++j)
					sum_term += alpha_val[i - 1][j] * ri_val[abs(i - j)]; 
           
				ki[i]= (ri_val[i] - sum_term) / ei_val[i - 1];
				alpha_val[i][i] = ki[i];
				for(j = 1; j < i; ++j)
                 alpha_val[i][j] = alpha_val[i - 1][j] - (ki[i] * alpha_val[i - 1][i - j]);
				ei_val[i] = (1 - (ki[i] * ki[i]))* ei_val[i - 1];
			}

			//printf("\n\n");
			for(i = 1; i < NO_OF_CEP + 1; ++i){
				ai_val[i] = alpha_val[11][i];
				//printf("ai_val[%d]: %lf\n", i, ai_val[i]);
			}
     
			ci_val[0] = log(ri_val[0]);
			//printf("\n\nci_val[0]: %lf\n",ci_val[0]);
			//fprintf(fp, "%lf\t", ci_val[0]);

			sum_term = 0;     

			for(m = 1; m < NO_OF_CEP + 1; ++m){
				sum_term = 0;
				for(k = 1; k <= m - 1; ++k)
                 sum_term += ((double)k * ci_val[k] * ai_val[m - k]) / (double)m;
				ci_val[m] = ai_val[m] + sum_term;
				fprintf(fp, "%.30lf\t", ci_val[m]);
				//printf("ci_val[%d]: %lf\n", m, ci_val[m]);
			}
			fprintf(fp, "\n");
		}
	 }
     fclose(fp);
}

     
     

/*void get_frame()
{
     int i;
     
     for(i = 0; i < SAMPLE_IN_FRAME; ++i){
           current_frame[i] = data[i + start];
           //printf("%lf ", current_frame[i]);
     }
     
     start += 80;
}*/

void apply_ham_win()
{
     int i;
     FILE* fp1 = fopen("new2.txt", "w");
     
     for(i = 0; i < SAMPLE_IN_FRAME; ++i){
           current_frame[i] *= ham_win[i];
           fprintf(fp1, "%lf\n", current_frame[i]);
     }
     fclose(fp1); 
}

void calculate_ri()
{
     int i, m;
     for(i = 0; i < NO_OF_CEP; ++i)
           ri_val[i] = 0;
     
     //printf("\n");   
     for(m = 0; m < NO_OF_CEP; ++m){
           for(i = 0; i < SAMPLE_IN_FRAME - m; ++i) 
                 ri_val[m] +=(current_frame[i] * current_frame[i + m]);
           //printf("ri_val[%d]: %lf\n", m, ri_val[m]);
     }       
}


void kmeans()
{
    char ch;
	int i, k;
	double c;
	double deviation = 100;
	FILE* fp = fopen(file_name7, "r");
	FILE* fp1 = fopen(file_name8, "r");
	if(!fp)
		exit(0);
	if(!fp1)
		exit(0);
	fprintf(fp_log, "Kmeans starts\n");
	/*
	while(!feof(fp)){
        fscanf(fp, "%c", &ch);
        if(ch == '\n')
              ++N;
    }
    ++N; 
     
    while(!feof(fp1)){
        fscanf(fp1, "%c", &ch);
        if(ch == '\n')
              ++K;
    }
    */
    fprintf(fp_log, "Size of codebook = %d, No. of training vectors = %d\n", K, N);
    
    //fseek(fp, 0, SEEK_SET);
    //fseek(fp1, 0, SEEK_SET);
 
    min_distance = (double* ) malloc(sizeof(double) * N);
	data_set = (double** ) malloc(sizeof(double*) * N);
	data_set_index = (int*) malloc(sizeof(int) * N);
	for(i=0;i < N; ++i){
        data_set[i]=(double *)malloc(sizeof(double) * NO_OF_CEP);
		for(k = 0; (k < NO_OF_CEP) && (!feof(fp)); ++k){
			fscanf(fp, "%lf", &c);
			data_set[i][k] = c;
		}
    }

	cluster_length = (int*) malloc(sizeof(int) * K );
    
    centroid = (double**) malloc(sizeof(double*) * K);
    for(i=0;i < K; ++i){
        centroid[i] = (double*) malloc(sizeof(double) * NO_OF_CEP );
        			
		for(k = 0; (k < NO_OF_CEP) && (!feof(fp1)); ++k){
			fscanf(fp1, "%lf", &c);
			//printf("%35.30lf\t", c);
			centroid[i][k] = c;
			//printf("%35.30lf\t", centroid[i][k]);
		}
    }
    for(i = 0; i < K; ++i)
        cluster_length[i] = 0;
        
    prev_distortion = 1000;
    i = 0;
    while(deviation > EPSILON){
        update_centroid();
        find_avg_distortion();
        deviation = prev_distortion - new_distortion;
		fprintf(fp_dev, "%lf\n", deviation); 
		//printf("deviation: %lf\n", deviation);
        prev_distortion = new_distortion;
        ++i;
    }
    fprintf(fp_log, "no of iterations needed:%d\n", i);
    fclose(fp1);
    fp1 = fopen(file_name8, "w");
    for(i = 0; i < K; ++i){
          for(k = 0; k < NO_OF_CEP; ++k)
                fprintf(fp1, "%0.30lf\t", centroid[i][k]);
          fprintf(fp1,"\n");
    }
    //fprintf(fp_log, "Kmeans ends\n");
	fclose(fp);
	fclose(fp1);
}

void update_centroid()
{
     int i1, i, j, k;
     double distance,min_distance1;
     
     for(k = 0; k < N; ++k){
          distance = 0;
          min_distance1 = 100;
          for(i = 0; i < K; ++i){
                //printf("min_distance: %lf\n", min_distance1);
                distance = 0;          
                for(j = 0; j < NO_OF_CEP; ++j)
                      distance += weights[j]*((data_set[k][j] - centroid[i][j]) * (data_set[k][j] - centroid[i][j]));
                if(distance < min_distance1){
                      i1 = i;
                      min_distance1 = distance;
                }
          }
          ++cluster_length[i1];
          min_distance[k] = min_distance1;
          data_set_index[k] = i1;
     }
     
     //printf("the classification is:\n");
     for(i = 0; i < K; ++i)
           fprintf(fp_log, "%d, ", cluster_length[i]);

     fprintf(fp_log,"\n");

     for(i = 0; i < K; ++i)
           for(j = 0; j < NO_OF_CEP; ++j)
                 centroid[i][j] = 0;
                 
     for(i = 0; i < N; ++i)
           for(j = 0; j < NO_OF_CEP; ++j)
                 centroid[data_set_index[i]][j] += data_set[i][j];
                 
     for(i = 0; i < K; ++i){
           for(j = 0; j < NO_OF_CEP; ++j)
                 centroid[i][j] /= (double)cluster_length[i];
           cluster_length[i] = 0;
     }
}

void find_avg_distortion()
{
     int i;
     new_distortion = 0;
     for(i = 0 ; i < N; ++i)
           new_distortion += min_distance[i];
     
     new_distortion /= (double)N;
     fprintf(fp_dist,"%0.30lf\n", new_distortion);
}

void LBG()
{
     int m;
     fp_dev = fopen(file_name9, "w");
     fp_dist = fopen(file_name12, "w");
     fp_log = fopen(file_name13, "w");
     
     initialize_LBG();
     while(K < CODEBOOK_SIZE){
             split_codebook();
             kmeans();
     }
     fclose(fp_dev);
     fclose(fp_dist);
     fclose(fp_log);
     printf("Done.\n");
}

void split_codebook()
{
     int i, j;
     char ch;
     double d;
     FILE* fp1 = fopen(file_name8, "r");
     FILE* fp2 = fopen(file_name10, "w");
     FILE* fp3 = fopen(file_name11, "w");
     
     //printf("In split_codebook beginning"); 
     fprintf(fp_log,"codebook size:%d\n", K);
     for(i = 0; i < K; ++i){
           for(j = 0; j < NO_OF_CEP; ++j){
                 fscanf(fp1, "%lf", &centroid1[j]);
                 fprintf(fp2, "%.30lf\t", centroid1[j] *(1 + EPSILON2));
                 fprintf(fp3, "%.30lf\t", centroid1[j] * (1 - EPSILON2));
           }
           fprintf(fp2,"\n");
           fprintf(fp3,"\n");
     }
     fclose(fp1);
     fclose(fp2);
     fclose(fp3);
     
     fp1 = fopen(file_name8, "w");
     fp2 = fopen(file_name10, "r");
     fp3 = fopen(file_name11, "r");
     //printf("temp_positive.txt\n");
     if(!feof(fp2)){
           for(i = 0; i < K; ++i){
                for(j = 0; j < NO_OF_CEP; ++j){             
                      fscanf(fp2, "%lf", &d);
                      fprintf(fp1, "%.30lf\t", d);
                      //printf("%.30lf\t", d);
                }
           fprintf(fp1, "\n", d);
           }
     }
     //printf("temp_negative.txt\n");
     if(!feof(fp3)){
           for(i = 0; i < K; ++i){
                 for(j = 0; j < NO_OF_CEP; ++j){            
                       fscanf(fp3, "%lf", &d);
                       fprintf(fp1, "%.30lf\t", d);
                       //printf("%.30lf\t", d);
                 }
                 fprintf(fp1, "\n", d); 
           }
     }
     fclose(fp1);
     fclose(fp2);
     fclose(fp3);
     K = K * 2;
     //printf("in split_codebook ending");
}
 
void initialize_LBG()
{
     int i, j;
     char ch;
     double d;
     FILE* fp = fopen(file_name7, "r");
     
     while(!feof(fp)){
        fscanf(fp, "%c", &ch);
        if(ch == '\n')
              ++N;
     }
     ++N;
     fclose(fp);
     fp = fopen(file_name7, "r");
     fprintf(fp_log, "No. of training vectors:%d\n", N);
     for(j = 0; j < NO_OF_CEP; ++j)
                 centroid1[j] = 0;
     
     for(i = 0; i < N; ++i){
           for(j = 0; j < NO_OF_CEP; ++j){
                 fscanf(fp, "%lf", &d);
                 //printf("%35.30lf\t", d);
                 centroid1[j] += d;
           }
     }
     fclose(fp);
     
     fp = fopen("code_book.txt", "w");
     for(j = 0; j < NO_OF_CEP; ++j){
                 centroid1[j] /= N;
                 //printf("%35.30lf\t", centroid1[j]); 
                 fprintf(fp, "%.30lf\t", centroid1[j]);
                 //fprintf(fp1, "%35.30lf\t", centroid1[j]);
     }
     fclose(fp);
     //printf("in intialize_LBG");    
}
     
void HMM()
{    
    int i, j;
    for(i = 0; i < N1; ++i){
          for(j = 0; j < M; ++j){
                B[i][j] = (double)1/(double)8;
                //printf("%lf", B[i][j]);
          }
    }
    DCshift();
    normalize();
    silence_removal();
    hamming_window_values();
    start = 0;
    calculate_ci();
          
    get_N();
    get_index();
    solution_prob1();
	solution_prob2();
	for(i = 0; i < 30; ++i)
	solution_prob3();
	fclose(fp_model);
}

void solution_prob1()
{
     int i, j, k, t;
     double sum = 0;
     P = 0;
     
	 for(i = 0; i < N; ++i){
		 for(j = 0; j < N1; ++j){
			 alpha[i][j] = 0;
			 beta[i][j] = 0;
          }
    }

     for(i = 0; i < N1; ++i){
                //printf("O[1]:%d,", O[1]);
                //printf("pi[%d]: %d, B:%lf\n",i, pi[i], B[i][O[1]]);                                        // initilization
                alpha[0][i] += pi[i] * B[i][O[1]];                  // k is o/p symbol index for o1 observation
                //printf("%0.20lf\n", alpha[1][i]);
     }
     
           
     for(t = 0; t < N - 1; ++t){                                         // Induction
           for(j = 0; j < N1; ++j){
               sum = 0;
               for(i = 0; i < N1; ++i){
                     //printf("%lf, %lf\n", alpha[t][i], A[i][j]);
                     sum += alpha[t][i] * A[i][j];
                     //printf("%0.20lf\n", sum);
                     }
               alpha[t + 1][j] = sum * B[j][O[t+1]];
               //printf("%0.100lf\n", alpha[t + 1][j]);
           }
     }
     
     for(i = 0; i < N1; ++i){                                        // termination
           P += alpha[N - 1][i];
           //printf("%0.100lf\n", P);
           beta[N - 1][i] = 1;                                           // initialization of beta
     }
     
     sum = 0;
     
     for(t = N - 2; t >= 0; --t){                                     // induction
           for(i = 0; i < N1; ++i){
                 for(j = 0; j < N1; ++j)
                       beta[t][i] += A[i][j] * B[j][O[t + 1]] * beta[t + 1][j];
                 //printf("%0.100lf\n", beta[t][i]);
           }
     }
}

void solution_prob2()
{
     int i, j, t, max_index;
     double max = 0, p1;
     
     for(i = 0; i < N1; ++i){
           delta[0][i] = pi[i] * B[i][O[0]];
           psi[0][i] = 0;                        // initialization
     }
     
     for(t = 1; t < N; ++t){                     // recursion
           for(j = 0; j < N1; ++j){
                 max = 0;
                 max_index = 0;
                 for(i = 0; i < N1; ++i){
                     p1 = delta[t - 1][i] * A[i][j];
                     if(p1 > max){
                           max = p1;
                           max_index = i;
                     }
                 }
                 delta[t][j] = max * B[j][O[t]];
                 psi[t][j] = max_index;
           }
     }
     
     for(i = 0; i < N1; ++i){                     // termination
           if(delta[N - 1][i] > Pstar){
                      Pstar = delta[N - 1][i];
                      Qstar[N - 1] = i;
           }
     }
     
     printf("%.100lf\n",Pstar);
     //Pstar_ten[Pstar_ten_index] = Pstar;
     
     for(t = N - 2; t>= 0; --t)
           Qstar[t] = psi[t + 1][Qstar[t + 1]];
     for(t = 0; t < N; ++t)
           printf("%d ", Qstar[t]);
}
void solution_prob3()
{
     int i, j, t, k;
     double sum = 0;
     fp_model = fopen(file_name15, "w");
     
     //printf("\nIn prob_solution3: \n");
     
     for(t = 0; t < N; ++t){
           sum = 0;
           for(i = 0; i < N1; ++i){
                 for(j = 0; j < N1; ++j){
                       sum += alpha[t][i] * A[i][j] * B[j][O[t + 1]] * beta[t + 1][j];
                 }
           }
           product[t] = sum;
     }
     
     //printf("Step 1: \n");
     
     for(t = 0; t < N - 1; ++t){
           sum = 0;
           for(i = 0; i < N1; ++i){
                 for(j = 0; j < N1; ++j){
                       xi[t][i][j] = (alpha[t][i] * A[i][j] * B[j][O[t + 1]] * beta[t + 1][j]) / product[t];
                       //printf("hello%0.50lfHello\n", xi[t][i][j]);
                 }
           }
     }
     //printf("Step2: \n");
     for(t = 0; t < N; ++t){
            for(i = 0; i < N1; ++i){
                  sum = 0;
                  for(j = 0; j < N1; ++j){
                        sum += xi[t][i][j];
                  }
                  gamma[t][i] = sum;
            }
     }
     //printf("Step3: \n");
     // re-estimation of A matrix
     for(i = 0; i < N1; ++i){
            for(j = 0; j < N1; ++j){
                  xi_sum[i][j] = 0;
                  for(t = 0; t < N - 1; ++t){
                      xi_sum[i][j] += xi[t][i][j];
                  }
            }
     }
     //printf("Step4: \n");
     for(i = 0; i < N1; ++i){
           gamma_sum[i] = 0;
           for(t = 0; t < N - 1; ++t){
                 gamma_sum[i] += gamma[t][i];
               //printf("%lf\n", gamma_sum[i]);
               //printf("i = %d, t = %d\n", i, t);
           }
     }
     //printf("Step5: \n");
     for(i = 0; i < N1; ++i){
           sum = 0;
           for(j = 0; j < N1; ++j){

                 A[i][j] = xi_sum[i][j] / gamma_sum[i];
                 sum += A[i][j];
                 fprintf(fp_model, "%0.50lf\n", A[i][j]);
           }
     }
     
     // re-estimation of B matrix
     for(i = 0; i < N1; ++i){
           gamma_sum[i] = 0;
           for(t = 0; t < N; ++t){
                 gamma_sum[i] += gamma[t][i];
               //printf("%lf\n", gamma_sum[i]);
               //printf("i = %d, t = %d\n", i, t);
           }
     }
     
     //printf("Step6: \n");
     for(j = 0; j < N1; ++j){
           for(k = 0; k < M; ++k){
                 sum = 0;
                 for(t = 0; t < N; ++t){
                       if(O[t] == k)
                               sum += gamma[t][j];
                 }
                 B[j][k] = sum / gamma_sum[j];
                 fprintf(fp_model, "%0.50lf\n", B[j][k]);
           }
     }
     
     for(j = 0; j < N1; ++j){
           sum = 0;
           for(k = 0; k < M; ++k){
                 sum += B[j][k];
                 //printf("sum is: %0.50lf\n", sum);
           }
           //printf("Sum :%0.50lf\n");
     }
     solution_prob2();
     fclose(fp_model);
}

void get_N()
{
     char ch;
     FILE* fp = fopen(file_name6, "r");
     FILE* fp1 = fopen(file_name8, "r");
     
     while(!feof(fp)){
        fscanf(fp, "%c", &ch);
        if(ch == '\n')
              ++N;
     }
     --N;
     while(!feof(fp1)){
        fscanf(fp1, "%c", &ch);
        if(ch == '\n')
              ++K;
    }
    printf("Size of codebook = %d, No. of training vectors = %d\n", K, N);
    fclose(fp);
    fclose(fp1);
}

void get_index()
{
    char ch;
	int i, k;
	double c;
	double deviation = 100;
	FILE* fp = fopen(file_name6, "r");
	FILE* fp1 = fopen(file_name8, "r");
	if(!fp)
		exit(0);
	if(!fp1)
		exit(0);
 
    min_distance = (double* ) malloc(sizeof(double) * N);
	data_set = (double** ) malloc(sizeof(double*) * N);
	data_set_index = (int*) malloc(sizeof(int) * N);
	for(i=0;i < N; ++i){
        data_set_index[i] = 0;
        min_distance[i] = 0;     
        data_set[i]=(double *)malloc(sizeof(double) * NO_OF_CEP);
		for(k = 0; (k < NO_OF_CEP) && (!feof(fp)); ++k){
			fscanf(fp, "%lf", &c);
			data_set[i][k] = c;
		}
    }

	cluster_length = (int*) malloc(sizeof(int) * K );
    
    centroid = (double**) malloc(sizeof(double*) * K);
    for(i=0;i < K; ++i){
        centroid[i] = (double*) malloc(sizeof(double) * NO_OF_CEP );
        			
		for(k = 0; (k < NO_OF_CEP) && (!feof(fp1)); ++k){
			fscanf(fp1, "%lf", &c);
			//printf("%35.30lf\t", c);
			centroid[i][k] = c;
			//printf("%35.30lf\t", centroid[i][k]);
		}
    }
    for(i = 0; i < K; ++i)
        cluster_length[i] = 0;
        
    prev_distortion = 1000;
    i = 0;
    calculate_index();
    //fprintf(fp_log, "Kmeans ends\n");
	fclose(fp);
	fclose(fp1);
}

void calculate_index()
{
     int i1, i, j, k;
     double distance,min_distance1;
     
     for(k = 0; k < N; ++k){
          distance = 0;
          min_distance1 = 100;
          for(i = 0; i < K; ++i){
                //printf("min_distance: %lf\n", min_distance1);
                distance = 0;          
                for(j = 0; j < NO_OF_CEP; ++j)
                      distance += weights[j]*((data_set[k][j] - centroid[i][j]) * (data_set[k][j] - centroid[i][j]));
                if(distance < min_distance1){
                      i1 = i;
                      min_distance1 = distance;
                }
                //printf("min_distance:%0.20lf, data_index:%d\n", min_distance1, i1);
          }
          ++cluster_length[i1];
          min_distance[k] = min_distance1;
          data_set_index[k] = i1;
          //printf("min_distance:%0.20lf, data_index:%d\n", min_distance[k], data_set_index[k]);
     }
     
     for(i = 0; i < N; ++i){
           O[i] = data_set_index[i];
           printf("O[%d]: %d\n",i, O[i]);
           //printf("min_distance %0.20lf\n", min_distance[i]);
     }
}

