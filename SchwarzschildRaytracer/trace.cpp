#include <random>
#include <array>
#include <algorithm>
#include <numeric>
#include <future>
#include <mutex>
#include <type_traits>
#include <SFML/Graphics.hpp>
#include <SFML/OpenGL.hpp>

static const double Pi = 3.1415926535897932384626433;
static const double deg = Pi / 180.0;

static const int diskWidth = 1024+512;
static const int diskHeight = 256*8;

static const int nths = std::thread::hardware_concurrency();

template<typename T> struct Vec
{
    T x, y, z;

    Vec():x{(T)0}, y{(T)0}, z{(T)0}{}
    Vec(T const& xx, T const& yy, T const& zz):x{xx}, y{yy}, z{zz}{}

    template<typename U> operator Vec<U>() const { return Vec<U>{(U)x, (U)y, (U)z}; }

    auto& operator+= (Vec<T> const& v){ x += v.x; y += v.y; z += v.z; return *this; }

    auto sum()const{ return x+y+z; }
    auto max()const{ return std::max({x, y, z}); }
    auto min()const{ return std::min({x, y, z}); }

    sf::Color sfColor() const 
    {
        auto r = x > (T)1 ? (T)1 : (x < (T)0 ? (T)0 : x);
        auto g = y > (T)1 ? (T)1 : (y < (T)0 ? (T)0 : y);
        auto b = z > (T)1 ? (T)1 : (z < (T)0 ? (T)0 : z);
        return sf::Color((int)(r*255), (int)(g*255), (int)(b*255));
    }
};

template<typename T> auto operator+( Vec<T> const& u, Vec<T> const& v ){ return Vec<T>{u.x+v.x, u.y+v.y, u.z+v.z}; }
template<typename T> auto operator-( Vec<T> const& u, Vec<T> const& v ){ return Vec<T>{u.x-v.x, u.y-v.y, u.z-v.z}; }
template<typename T, typename U> auto operator*( Vec<T> const& u, U      const& c ){ return Vec<T>{(T)(u.x*c), (T)(u.y*c), (T)(u.z*c)}; }
template<typename T, typename U> auto operator*( U      const& c, Vec<T> const& u ){ return Vec<T>{(T)(u.x*c), (T)(u.y*c), (T)(u.z*c)}; }
template<typename T, typename U> auto operator/( Vec<T> const& u, U      const& c ){ return Vec<T>{(T)(u.x/c), (T)(u.y/c), (T)(u.z/c)}; }

template<typename T> auto dot( Vec<T> const& u, Vec<T> const& v ){ return u.x*v.x + u.y*v.y + u.z*v.z; }
template<typename T> auto cross( Vec<T> const& u, Vec<T> const& v ){ return Vec<T>{u.y*v.z-u.z*v.y, u.z*v.x-u.x*v.z, u.x*v.y-u.y*v.x}; }
template<typename T> auto length( Vec<T> const& u ){ return sqrt(dot(u, u)); }
template<typename T> auto sqlength( Vec<T> const& u ){ return dot(u, u); }
template<typename T> auto distance( Vec<T> const& u, Vec<T> const& v ){ return sqrt(sq(u.x-v.x)+sq(u.y-v.y)+sq(u.z-v.z)); }
template<typename T> auto angle( Vec<T> const& u, Vec<T> const& v ){ return std::acos( dot(u, v) / (length(u)*length(v) ) ); }

template<typename T> auto rot_around( Vec<T> const& v, Vec<T> const& axis, T const& theta )
{
    auto C = cos(theta);
    auto S = sin(theta);
    return v*C + cross(axis, v)*S + axis*dot(axis, v)*((T)1 - C);
}

template<typename T> auto xzortho( Vec<T> const& v ){ return Vec<T>{v.z, v.y, -v.x}; }
template<typename T> auto normalize( Vec<T> const& v ){ return v / length(v); }

/*template<typename T>
void normalize(T& x, T& y, T& z)
{
    auto r = sqrt(sq(x)+sq(y)+sq(z));
    x /= r; y /= r; z /= r;
}*/

template<typename T> auto gamma_correct( Vec<T> const& rgb, T const& gamma = 2.2)
{
    auto f = [&](auto const& x){ return x < 0.0031308 ? 12.92*x : (1+0.055)*pow(x, 1.0/2.4) - 0.055; };
    auto res = rgb;
    res.x = (T)f(res.x);
    res.y = (T)f(res.y);
    res.z = (T)f(res.z);
    return res;
}

template<typename T> auto redistribute_rgb( Vec<T> const& rgb )
{
    static_assert( std::is_floating_point<T>::value, "Error: redistribute_rgb argument typpe is not floating point");
    auto max = rgb.max();
    if( max <= (T)1 ){ return rgb; }
    
    auto sum = rgb.sum();
    if( sum >= (T)3 ){ return Vec<T>{(T)1, (T)1, (T)1}; }

    auto q = ((T)3 - sum) / ((T)3*max - sum);
    auto gray = (T)1 - q*max;
    return Vec<T>{ gray + q*rgb.x, gray + q*rgb.y, gray + q*rgb.z };
}


//XYZ color matching functions 
std::vector<double> cmf =
{
360, 0.000129900000, 0.000003917000, 0.000606100000,
365, 0.000232100000, 0.000006965000, 0.001086000000,
370, 0.000414900000, 0.000012390000, 0.001946000000,
375, 0.000741600000, 0.000022020000, 0.003486000000,
380, 0.001368000000, 0.000039000000, 0.006450001000,
385, 0.002236000000, 0.000064000000, 0.010549990000,
390, 0.004243000000, 0.000120000000, 0.020050010000,
395, 0.007650000000, 0.000217000000, 0.036210000000,
400, 0.014310000000, 0.000396000000, 0.067850010000,
405, 0.023190000000, 0.000640000000, 0.110200000000,
410, 0.043510000000, 0.001210000000, 0.207400000000,
415, 0.077630000000, 0.002180000000, 0.371300000000,
420, 0.134380000000, 0.004000000000, 0.645600000000,
425, 0.214770000000, 0.007300000000, 1.039050100000,
430, 0.283900000000, 0.011600000000, 1.385600000000,
435, 0.328500000000, 0.016840000000, 1.622960000000,
440, 0.348280000000, 0.023000000000, 1.747060000000,
445, 0.348060000000, 0.029800000000, 1.782600000000,
450, 0.336200000000, 0.038000000000, 1.772110000000,
455, 0.318700000000, 0.048000000000, 1.744100000000,
460, 0.290800000000, 0.060000000000, 1.669200000000,
465, 0.251100000000, 0.073900000000, 1.528100000000,
470, 0.195360000000, 0.090980000000, 1.287640000000,
475, 0.142100000000, 0.112600000000, 1.041900000000,
480, 0.095640000000, 0.139020000000, 0.812950100000,
485, 0.057950010000, 0.169300000000, 0.616200000000,
490, 0.032010000000, 0.208020000000, 0.465180000000,
495, 0.014700000000, 0.258600000000, 0.353300000000,
500, 0.004900000000, 0.323000000000, 0.272000000000,
505, 0.002400000000, 0.407300000000, 0.212300000000,
510, 0.009300000000, 0.503000000000, 0.158200000000,
515, 0.029100000000, 0.608200000000, 0.111700000000,
520, 0.063270000000, 0.710000000000, 0.078249990000,
525, 0.109600000000, 0.793200000000, 0.057250010000,
530, 0.165500000000, 0.862000000000, 0.042160000000,
535, 0.225749900000, 0.914850100000, 0.029840000000,
540, 0.290400000000, 0.954000000000, 0.020300000000,
545, 0.359700000000, 0.980300000000, 0.013400000000,
550, 0.433449900000, 0.994950100000, 0.008749999000,
555, 0.512050100000, 1.000000000000, 0.005749999000,
560, 0.594500000000, 0.995000000000, 0.003900000000,
565, 0.678400000000, 0.978600000000, 0.002749999000,
570, 0.762100000000, 0.952000000000, 0.002100000000,
575, 0.842500000000, 0.915400000000, 0.001800000000,
580, 0.916300000000, 0.870000000000, 0.001650001000,
585, 0.978600000000, 0.816300000000, 0.001400000000,
590, 1.026300000000, 0.757000000000, 0.001100000000,
595, 1.056700000000, 0.694900000000, 0.001000000000,
600, 1.062200000000, 0.631000000000, 0.000800000000,
605, 1.045600000000, 0.566800000000, 0.000600000000,
610, 1.002600000000, 0.503000000000, 0.000340000000,
615, 0.938400000000, 0.441200000000, 0.000240000000,
620, 0.854449900000, 0.381000000000, 0.000190000000,
625, 0.751400000000, 0.321000000000, 0.000100000000,
630, 0.642400000000, 0.265000000000, 0.000049999990,
635, 0.541900000000, 0.217000000000, 0.000030000000,
640, 0.447900000000, 0.175000000000, 0.000020000000,
645, 0.360800000000, 0.138200000000, 0.000010000000,
650, 0.283500000000, 0.107000000000, 0.000000000000,
655, 0.218700000000, 0.081600000000, 0.000000000000,
660, 0.164900000000, 0.061000000000, 0.000000000000,
665, 0.121200000000, 0.044580000000, 0.000000000000,
670, 0.087400000000, 0.032000000000, 0.000000000000,
675, 0.063600000000, 0.023200000000, 0.000000000000,
680, 0.046770000000, 0.017000000000, 0.000000000000,
685, 0.032900000000, 0.011920000000, 0.000000000000,
690, 0.022700000000, 0.008210000000, 0.000000000000,
695, 0.015840000000, 0.005723000000, 0.000000000000,
700, 0.011359160000, 0.004102000000, 0.000000000000,
705, 0.008110916000, 0.002929000000, 0.000000000000,
710, 0.005790346000, 0.002091000000, 0.000000000000,
715, 0.004109457000, 0.001484000000, 0.000000000000,
720, 0.002899327000, 0.001047000000, 0.000000000000,
725, 0.002049190000, 0.000740000000, 0.000000000000,
730, 0.001439971000, 0.000520000000, 0.000000000000,
735, 0.000999949300, 0.000361100000, 0.000000000000,
740, 0.000690078600, 0.000249200000, 0.000000000000,
745, 0.000476021300, 0.000171900000, 0.000000000000,
750, 0.000332301100, 0.000120000000, 0.000000000000,
755, 0.000234826100, 0.000084800000, 0.000000000000,
760, 0.000166150500, 0.000060000000, 0.000000000000,
765, 0.000117413000, 0.000042400000, 0.000000000000,
770, 0.000083075270, 0.000030000000, 0.000000000000,
775, 0.000058706520, 0.000021200000, 0.000000000000,
780, 0.000041509940, 0.000014990000, 0.000000000000,
785, 0.000029353260, 0.000010600000, 0.000000000000,
790, 0.000020673830, 0.000007465700, 0.000000000000,
795, 0.000014559770, 0.000005257800, 0.000000000000,
800, 0.000010253980, 0.000003702900, 0.000000000000,
805, 0.000007221456, 0.000002607800, 0.000000000000,
810, 0.000005085868, 0.000001836600, 0.000000000000,
815, 0.000003581652, 0.000001293400, 0.000000000000,
820, 0.000002522525, 0.000000910930, 0.000000000000,
825, 0.000001776509, 0.000000641530, 0.000000000000,
830, 0.000001251141, 0.000000451810, 0.000000000000
};

//The following code is based on: http://www.fourmilab.ch/documents/specrend/specrend.c
struct colourSystem {
    char *name;     	    	    /* Colour system name */
    double xRed, yRed,	    	    /* Red x, y */
           xGreen, yGreen,  	    /* Green x, y */
           xBlue, yBlue,    	    /* Blue x, y */
           xWhite, yWhite,  	    /* White point x, y */
	   gamma;   	    	    /* Gamma correction for system */
};

#define IlluminantC     0.3101, 0.3162	    	/* For NTSC television */
#define IlluminantD65   0.3127, 0.3291	    	/* For EBU and SMPTE */
#define IlluminantE 	0.33333333, 0.33333333  /* CIE equal-energy illuminant */
#define GAMMA_REC709	709		                /* Rec. 709 */

static struct colourSystem
                  /* Name                  xRed    yRed    xGreen  yGreen  xBlue  yBlue    White point        Gamma   */
    NTSCsystem  =  { "NTSC",               0.67,   0.33,   0.21,   0.71,   0.14,   0.08,   IlluminantC,    GAMMA_REC709 },
    EBUsystem   =  { "EBU (PAL/SECAM)",    0.64,   0.33,   0.29,   0.60,   0.15,   0.06,   IlluminantD65,  GAMMA_REC709 },
    SMPTEsystem =  { "SMPTE",              0.630,  0.340,  0.310,  0.595,  0.155,  0.070,  IlluminantD65,  GAMMA_REC709 },
    HDTVsystem  =  { "HDTV",               0.670,  0.330,  0.210,  0.710,  0.150,  0.060,  IlluminantD65,  GAMMA_REC709 },
    CIEsystem   =  { "CIE",                0.7355, 0.2645, 0.2658, 0.7243, 0.1669, 0.0085, IlluminantE,    GAMMA_REC709 },
    Rec709system = { "CIE REC 709",        0.64,   0.33,   0.30,   0.60,   0.15,   0.06,   IlluminantD65,  GAMMA_REC709 };

template<typename T>
void xyz_to_rgb(struct colourSystem &cs,
                T xc, T yc, T zc,
                T &r, T &g, T &b)
{
    T xr, yr, zr, xg, yg, zg, xb, yb, zb;
    T xw, yw, zw;
    T rx, ry, rz, gx, gy, gz, bx, by, bz;
    T rw, gw, bw;

    xr = (T)cs.xRed;    yr = (T)cs.yRed;    zr = 1 - (xr + yr);
    xg = (T)cs.xGreen;  yg = (T)cs.yGreen;  zg = 1 - (xg + yg);
    xb = (T)cs.xBlue;   yb = (T)cs.yBlue;   zb = 1 - (xb + yb);

    xw = (T)cs.xWhite;  yw = (T)cs.yWhite;  zw = 1 - (xw + yw);

    /* xyz -> rgb matrix, before scaling to white. */
    
    rx = (yg * zb) - (yb * zg);  ry = (xb * zg) - (xg * zb);  rz = (xg * yb) - (xb * yg);
    gx = (yb * zr) - (yr * zb);  gy = (xr * zb) - (xb * zr);  gz = (xb * yr) - (xr * yb);
    bx = (yr * zg) - (yg * zr);  by = (xg * zr) - (xr * zg);  bz = (xr * yg) - (xg * yr);

    /* White scaling factors.
       Dividing by yw scales the white luminance to unity, as conventional. */
       
    rw = ((rx * xw) + (ry * yw) + (rz * zw)) / yw;
    gw = ((gx * xw) + (gy * yw) + (gz * zw)) / yw;
    bw = ((bx * xw) + (by * yw) + (bz * zw)) / yw;

    /* xyz -> rgb matrix, correctly scaled to white. */
    
    rx = rx / rw;  ry = ry / rw;  rz = rz / rw;
    gx = gx / gw;  gy = gy / gw;  gz = gz / gw;
    bx = bx / bw;  by = by / bw;  bz = bz / bw;

    /* rgb of the desired point */
    
    r = (rx * xc) + (ry * yc) + (rz * zc);
    g = (gx * xc) + (gy * yc) + (gz * zc);
    b = (bx * xc) + (by * yc) + (bz * zc);
}

template<typename T>
Vec<T> mono_wavelength_to_xyz(T const& l)
{
    if( l < 360 || l > 829.999){ return Vec<T>{(T)0, (T)0, (T)0}; }

    auto q = std::fmod(l-360.0, 5.0)/5.0;
    auto i = (int)((l - 360.0)/5.0);
    int k = i+1;
    auto X = cmf[k*4+1]*q + cmf[i*4+1]*(1.0-q);
    auto Y = cmf[k*4+2]*q + cmf[i*4+2]*(1.0-q);
    auto Z = cmf[k*4+3]*q + cmf[i*4+3]*(1.0-q);
    return Vec<T>{(T)X, (T)Y, (T)Z};
    //auto R = +0.41847000 * X - 0.1586600 * Y - 0.082835 * Z;
    //auto G = -0.09116900 * X + 0.2524300 * Y + 0.015708 * Z;
    //auto B = +0.00092090 * X - 0.0025498 * Y + 0.178600 * Z;
}

template<typename T>
Vec<T> convert_xyz_to_rgb( Vec<T> const& c )
{
    colourSystem& cs = HDTVsystem;
    T R, G, B;
    auto csum = c.x + c.y + c.z;
    auto xyz = c / csum;
    xyz_to_rgb(cs, xyz.x, xyz.y, xyz.z, R, G, B);
    if(R<0){R=(T)0;}
    if(G<0){G=(T)0;}
    if(B<0){B=(T)0;}
    auto q = (T)1;//std::max(R, std::max(G, B));
    return Vec<T>{R/q, G/q, B/q};

   /* Vec<T> res;
    res.x = (T)( 3.240479 * c.x - 1.537150 * c.y - 0.498535 * c.z);
    res.y = (T)(-0.969256 * c.x + 1.875992 * c.y + 0.041556 * c.z);
    res.z = (T)( 0.055648 * c.x - 0.204043 * c.y + 1.057311 * c.z);
    if(res.x < 0){ res.x = (T)0; }
    if(res.y < 0){ res.y = (T)0; }
    if(res.z < 0){ res.z = (T)0; }
    return res;*/
}

//T in Kelvin, lambda in nm
auto planck_curve(double T, double lambda)
{
    static const double hcc2 = 1.19104295e-16;
    static const double hcdivkB = 0.0143877735*1e9;
    auto x = hcdivkB/(lambda*T);
    auto e = std::exp(x);
    return hcc2 / (std::pow(lambda*1e-9, 5.0) * (e-1.0)) * 1e-14;
}

template<typename T>
auto black_body_xyz( T const& Temp )
{
    Vec<float> sum{(float)0, (float)0, (float)0};
    for(float l = 360; l <= 830; l+=5.0)
    {
        auto t = mono_wavelength_to_xyz(l);
        auto A = planck_curve(Temp, l);
        sum.x += (float)(t.x*A);
        sum.y += (float)(t.y*A);
        sum.z += (float)(t.z*A);
    }
    return sum;//convert_xyz_to_rgb(sum);
}

auto sq   = [](auto const& x){ return x*x; };
auto cube = [](auto const& x){ return x*x*x; };

struct Params
{
    double M;
    double G;
    double fovw;//degrees
    double zcam;
    double ycam;
    double T0disk;//Kelvin
};

template<typename T> struct ThetaPhi
{
    T theta, phi;
};

struct Star
{
    ThetaPhi<float> angles;
    Vec<float> colxyz;
    float d, T;
};

const int blocksize = 16;
struct TraceParams
{
    int bx, by;
    int w, h;
    bool* finish;

    void reset()
    {
        //const auto q = h/blocksize;
        by = 0;//w/(blocksize*3);
        bx = 0;
    }

    void step()
    {
        if(!*finish)
        {
            const auto p = w/blocksize;
            const auto q = h/blocksize;

            if(by > q){ by = 0; *finish = true; }
            if(bx > p){ bx = 0; by += 1;}else{ bx += 1; }
        }
    }
};

void trace_n(TraceParams& tpar, std::vector<Star> const& stars, std::vector<std::vector<int>> const& starmap, std::vector<Vec<float>>& xyz, std::vector<sf::Color>& res, std::vector<float> const& disk, Params const& params)
{
    std::random_device rd;
    std::mutex mxyz;

    const double rs = 2.; //Schwarzschild radius
    const double robs = length(Vec<double>{0.0, params.ycam, params.zcam}); //Distance of observer
    const double rho_obs = sqrt(1. - rs/robs);
    const double da = params.fovw / (double)tpar.w;

    const auto rdin  = 3.00*rs; //Inner radius of disk in Schwarzschild units
    const auto rdout = 8.00*rs; //Outer radius of disk in Schwarzschild units
    const auto hdisk = 0.05*rs; //Disk thickness in Schwarzschild units

    const auto up = Vec<double>{0.0, 1.0, 0.0};

    const Vec<double> bh0 = {0.0, 0.0, 0.0};//{+1.5, 0.0, +3.15};
    const Vec<double> bh1 = {-1.5, 0.0, -3.15};

    //Disk velocity vector at location
    auto vel_at = [&](Vec<double> const& p)
    {
        auto b1 = p - bh0;
        auto b2 = p - bh1;
        auto R1 = distance(p, bh0);
        auto R2 = distance(p, bh1);
        auto V1 = sqrt(params.M*params.G / (R1 - rs));
        auto V2 = sqrt(params.M*params.G / (R2 - rs));

        return xzortho(b1)/length(b1)*V1;

        //auto q = R1 / (R1+R2);
        //return xzortho(b1)/length(b1)*V1 * (1.-q) + xzortho(b2)/length(b2)*V2 * q;
    };

    auto trace_block = [&](int bx, int by)
    {
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-0.5, 0.5);
        std::uniform_real_distribution<> disbl(0, blocksize);

        Vec<float> tmpxyz[(50+blocksize)*(50+blocksize)];
        Vec<float> color;
        float color_alpha;
        
        for(double bx0=0.0; bx0<blocksize; bx0 += 1.0/2.0)
        {
            for(double by0=0.0; by0<blocksize; by0 += 1.0/2.0)
            {
                //double x00 = disbl(gen), y00 = disbl(gen);
                double x0 = bx0;
                double y0 = by0;

                //Multisampling
                //int nr = 0;
                //while(nr < 1)
                //{
                    //generate pixel of view:
                    //double x0 = x00+dis(gen), y0 = y00+dis(gen);
        
                    //Horizontal, Vertical view angle
                    auto wx = (bx * blocksize + x0 - tpar.w/2) * da * deg;
                    auto wy = (by * blocksize + y0 - tpar.h/2) * da * deg;
        
                    //Initial position, velocity and direction vectors
                    auto r0 = Vec<double>{0.0,     params.ycam, params.zcam};
                    auto v00 = Vec<double>{sin(wx), sin(wy),     -1.0};
                    auto nv0 = normalize(v00);
                    auto nr0 = normalize(r0);

                    //Direction vectors for changing into orbital plane determined by the radius and velocity vectors
                    auto n = normalize(cross(nr0, nv0)); //normal of movement plane
                    auto d = nr0;                        //radial direction
                    auto d0 = normalize(cross(up, n));   //radial direction in the plane of the disk
                    auto o = normalize(cross(n, d));     //transverse direction
                
                    //Initial values:
                    auto R = length(r0);
                    const auto Rlim = R * 1.1;
                    auto e0 = length(v00)*sqrt(1.0 - rs / R);//energy

                    auto Rdot0 = dot(nr0, v00);//radial velocity
                    auto h0 = length(cross(r0, v00));//angular momentum
    
                    auto Rdot = Rdot0;
                    auto lambda0 = 0.07;//affine parameter step

                    int k = 0;
                    int klim = (int)(100000);//step limiter
                    color = Vec<float>{0.f, 0.f, 0.f};
                    color_alpha = 1.0f;
                    bool hit = false;
                
                    //calculate phi0 (initial angle, measured from the disk):
                    //auto phi0 = 0.0;
                    /*{
                        auto q1 = std::abs(dot(r0,  R * d));
                        auto q2 = std::abs(dot(r0, -R * d));

                        phi0 = q1 < q2 ? angle(r0, R*d) : angle(r0, -R*d);
                    }*/

                    auto phi = 0.0;
                    auto last = r0;
                    auto dl = 0.0;
                    Vec<double> rrn;
                    while(R < Rlim && k < klim)
                    {
                        auto Rdotl = Rdot;
                        double lambda = lambda0;
                        auto rho = sqrt(1.0 - rs / R);

                        if( R < rs ){ k = klim; hit = true; break; }

                        {
                            auto rr = rot_around(r0, n, phi);
                            auto nrr = normalize(rr);
                            last = rrn;
                            rrn = nrr*R;
                            auto x = rrn.x;
                            auto y = rrn.y;
                            auto z = rrn.z;
                            auto plane_sqr = sq(x)+sq(z);
                            dl = length(rrn-last);

                            if( R < 3.0 * rs ){ lambda /= 8.0; }
                            else if( R < 2.5 * rs ){ lambda /= 16.0; /*if(abs(lasty) < 0.1*rs){ lambda /= 55.0; }else{ lambda /= 25.0; }*/ }

                            if( sq(rdin) <= plane_sqr && plane_sqr <= sq(rdout) && sq(y) <= sq(4.5*hdisk) )
                            {
                                lambda = lambda0 / 24.0;

                                //sample disk density:
                                auto disk_angle = (int)((Pi + atan2(x, z))/(2.0*Pi)*diskHeight);
                                auto disk_rad   = (int)((sqrt(plane_sqr) - rdin)/(rdout - rdin)*diskWidth);
                                if(disk_rad >= diskWidth){ disk_rad = diskWidth-1; }
                                if(disk_angle >= diskHeight){ disk_angle = diskHeight-1; }
                                disk_rad = diskWidth-1 - disk_rad;
                                auto density = 1.0f*(float)dl*disk[disk_rad*diskHeight+disk_angle]*exp(-sq((float)y/(float)hdisk));

                                auto dvel = vel_at(Vec<double>{x, y, z});
                                auto sqmag = sqlength(dvel);
                                auto mag = sqrt(sqmag);

                                auto gamma = 1.0 / sqrt(1.0 - sqmag);

                                auto T = /*mag * */params.T0disk;
                       
                                auto KT = normalize(rot_around(o, n, phi)) * h0 / R;

                                auto q = dot(dvel, KT)/(gamma*rho);
                            
                                //Doppler factor = z + 1 due to motion
                                auto D = gamma * (1 - q);
                                //Doppler factor due to gravitational well
                                auto Dz = rho_obs / rho;

                                double Flux = 1.0;
                                {
                                    auto rm = R/params.M;
                                    auto S = sqrt(rm);
                                    auto sqrt3 = sqrt(3.0);
                                    auto sqrt6 = sqrt(6.0);
                                    Flux = 3.0 * params.M * 0.01 / (8.0*Pi*(rm-3.0)*S*sq(rm)) * ( S - sqrt6 + sqrt3/3.0*std::log( (S + sqrt3) * (sqrt6 - sqrt3) / ( (S - sqrt3) * (sqrt6 + sqrt3) ) ) );
                                }

                                auto colmin = color.min();
                                color = color + color_alpha*black_body_xyz( (float)(T * (Dz * D) ) )*Flux*density;

                                //auto beta = 
                                color_alpha *= /*(1.0f - 4.0f*dl/hdisk)**/0.99965f*(1.0f-sqrt(density));//0.99

                                if(color_alpha <= 0.0005f){ color_alpha = 0.0005f; hit = true; break; }
                            }
                            
                        }


                        phi  += lambda * h0 / sq(R);
                        Rdot += lambda * (params.M / sq(R*rho) * (sq(Rdotl) - sq(e0)) + sq(h0*rho)/cube(R));
                        R    += lambda * Rdotl;

                        k += 1;
                    }
                    if( !hit )
                    {
                        auto rr = rot_around(r0, n, phi);
                        auto nrr = normalize(rr);
                        auto rrn = nrr*R;
                        auto x = rrn.x;
                        auto y = rrn.y;
                        auto z = rrn.z;

                        /*auto x = P.x;
                        auto y = P.y;
                        auto z = P.z;*/

                        auto r = sqrt(sq(x)+sq(y)+sq(z));
                        x /= r; y /= r; z /= r;

                        auto theta_tmp = acos(z);
                        auto phi_tmp = Pi + atan2(y, x);
                        auto tr = (int)(acos(z)/deg);
                        auto pr = 180+(int)(atan2(y, x)/deg);
                        auto mi = pr*180+tr;
                        double dmin = 100000.0;
                        int ss = -1;
                        auto const& starblock = starmap[mi];
                        for(int s=0; s<(int)starblock.size(); ++s)
                        {
                            auto const& star = stars[starblock[s]];
                            auto theta = (double)star.angles.theta;
                            auto phi   = (double)star.angles.phi;
                            auto xs = sin(theta)*cos(Pi+phi);
                            auto ys = sin(theta)*sin(Pi+phi);
                            auto zs = cos(theta);

                            auto d = sq(x-xs)+sq(y-ys)+sq(z-zs);
                            if(d < dmin)
                            {
                                ss = s; dmin = d;
                            }
                        }

                        if(ss >= 0)
                        {
                            auto const& star = stars[starblock[ss]];
                            if(dmin < star.d)
                            {
                                if(color_alpha > 0.0005f)
                                {
                                    color = color + 1e-9f*pow(color_alpha, 10.0f)*star.colxyz;
                                }
                                else
                                {
                                    color = star.colxyz;
                                }
                            }
                        }
                    }

                    auto Lo = [](auto const& x, auto const& w){ return 1.0 / (w*(1.0 + sq(sq(x/w)))); };

                    if( length(color) != 0.0f )
                    {
                        auto cs = (color.x+color.y+color.z);
                        for(int yi=-20; yi<=20; ++yi)
                        {
                            for(int xi=-20; xi<=20; ++xi)
                            {
                                auto xp = (int)(25+x0+xi);
                                auto yp = (int)(25+y0+yi);
                                auto Ex = -(sq(xi)+sq(yi))/2.0;
                                //auto A = (1.0 + pow(-Ex*(cs / 5e-3), 0.2));
                                auto dec = exp(Ex / sq(4.0));
                                auto ratx = (float)(dec*Lo(Ex, 0.05*0.74));//(A*exp(Ex/sq(0.25*0.74)));
                                auto raty = (float)(dec*Lo(Ex, 0.05*0.77));//(A*exp(Ex/sq(0.25*0.77)));
                                auto ratz = (float)(dec*Lo(Ex, 0.05*0.89));//(A*exp(Ex/sq(0.25*0.89)));
                                if( xp >= 0 && yp >= 0 && xp < blocksize+50 && yp < blocksize+50 )
                                {
                                    auto c0 = tmpxyz[yp*(blocksize+50)+xp];
                                    c0.x += color.x * ratx;
                                    c0.y += color.y * raty;
                                    c0.z += color.z * ratz;
                                    tmpxyz[yp*(blocksize+50)+xp] = c0;
                                }
                            }
                        }
                
                    }
                    //nr += 1;
                //} //Multisample
            }//block y
        }//block x

        int y0 = by*blocksize - 25;
        int x0 = bx*blocksize - 25;
        {
            std::lock_guard<std::mutex> lock(mxyz);
            for(int yi=0; yi<blocksize+50; ++yi)
            {
                for(int xi=0; xi<blocksize+50; ++xi)
                {
                    auto xp = x0 + xi;
                    auto yp = y0 + yi;
                    if( xp >= 0 && yp >= 0 && xp < tpar.w && yp < tpar.h )
                    {
                        auto c0 = xyz[yp*tpar.w+xp];
                        c0 = c0 + tmpxyz[yi * (blocksize+50) + xi];
                        xyz[yp*tpar.w+xp] = c0;
                    }
                }
            }
        }
    };//trace block

    std::vector<std::future<void>> threads(nths);
    for(auto& th : threads){ th = std::async( std::launch::async, trace_block, tpar.bx, tpar.by); tpar.step(); }
    for(auto& th : threads){ (void)th.get(); }
   
    /*auto f1 = std::async( std::launch::async, trace_block, tpar.bx, tpar.by);
    tpar.step();
    auto f2 = std::async( std::launch::async, trace_block, tpar.bx, tpar.by);
    tpar.step();
    auto f3 = std::async( std::launch::async, trace_block, tpar.bx, tpar.by);
    tpar.step();
    auto f4 = std::async( std::launch::async, trace_block, tpar.bx, tpar.by);
    tpar.step();*/

    /*(void)f1.get();
    (void)f2.get();
    (void)f3.get();
    (void)f4.get();*/
}

std::mt19937 gen(42);

auto gen_random_value_in_interval = [](auto const& lo, auto const& hi)
{
    std::uniform_real_distribution<> dis(lo, hi);
    return dis(gen);
};

auto gen_random_value_around = [](auto const& mu, auto const& sigma)
{
    std::lognormal_distribution<> dis(mu, sigma);
    return dis(gen);
};

auto gen_random_value_exp = [](auto const& lambda)
{
    std::exponential_distribution<> dis(lambda);
    return dis(gen);
};

auto gen_random_theta_phi = []
{
    std::uniform_real_distribution<> dis(0.0, 1.0);
    return ThetaPhi<double>{std::acos(1.0-2.0*dis(gen)), 2.0*Pi*dis(gen)};
};

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR pCmdLine, int nCmdShow)
{
    int width = 800, height = 800;
    sf::RenderWindow window(sf::VideoMode(width, height), "Gravitation Raytrace");

	sf::ContextSettings settings = window.getSettings();

	sf::RectangleShape rect(sf::Vector2f(width*1.0f, height*1.0f));
	rect.setFillColor(sf::Color(255, 255, 255));
	rect.setOutlineColor(sf::Color(255, 255, 255));

	sf::Font font;
	if (!font.loadFromFile("C:\\Windows\\Fonts\\arial.ttf")){ return -1; }

	sf::Text text;
	text.setFont(font); // font is a sf::Font
	text.setPosition(30, 30);
	text.setString("0");
	text.setCharacterSize(12);
	text.setFillColor(sf::Color(0, 255, 0));

    std::vector<Star> stars(100000);
    std::vector<std::vector<int>> starmap(360*180);

    for(int i=0; i<stars.size(); ++i)
    {
        float T;
        Vec<float> bb;
        auto dg = 1.0;
        bool accepted = false;
        while(!accepted)
        {
            const auto Tmag = 3000.0f;
            const auto spread = 0.7f;
            auto TI = 1.0f;
            auto giant = gen_random_value_in_interval(0.0, 100.0);
            auto var = 10000.0f*pow(10.0f, (float)gen_random_value_in_interval(0.5f, 10.5f));
            dg = 1.0;
            if( giant > 98.9 )
            {
                //Giant
                T = (float)(Tmag*gen_random_value_around(0.0, spread));
                TI = 1e5f * var;
                dg = 2.8;
            }
            else if(giant > 95.0)
            {
                T = (float)(Tmag*gen_random_value_around(0.0, spread));
                TI = 5e5f * var;
                dg = 2.1;
            }
            else if(giant > 92.0)
            {
                T = (float)(Tmag*gen_random_value_around(0.0, spread));
                TI = 2e3f * var;
                dg = 1.25;
            }
            else
            {
                //regular star
                T = (float)(Tmag*gen_random_value_around(0.0, spread));
                TI = var;
            }
            
            auto x = gen_random_value_in_interval(0.1f, 1500000.f);
            auto y = gen_random_value_in_interval(0.1f, 1500000.f);
            auto z = gen_random_value_in_interval(0.1f, 1500000.f);
            auto sqr = sq(x)+sq(y)+sq(z);

            bb = black_body_xyz(T)*TI;
            if( gen_random_value_in_interval(-1.0, 1.0) > 0.0 )
            {
                //binary partner
                auto T2 = (float)(Tmag*gen_random_value_around(0.0, spread));
                bb = bb + black_body_xyz(T2)*pow(10.0f, (float)gen_random_value_in_interval(0.5f, 10.5f));
            }
            bb = bb / sqr;
            if( bb.sum() > 1e-10 ){ accepted = true; }
        }
        
        auto as = gen_random_theta_phi();
        stars[i].angles.theta = (float)as.theta;
        stars[i].angles.phi = (float)as.phi;

        auto th = (int)(as.theta/deg);
        auto ph = (int)(as.phi/deg);

        auto helper = [&](int t, int p, int idx) mutable { if(p >= 0 && p <= 359 && t >= 0 && t <= 179){ starmap[p*180+t].push_back(idx); } };
        helper(th, ph-1, i);
        helper(th, ph  , i);
        helper(th, ph+1, i);
        helper(th+1, ph-1, i);
        helper(th+1, ph  , i);
        helper(th+1, ph+1, i);
        helper(th-1, ph-1, i);
        helper(th-1, ph  , i);
        helper(th-1, ph+1, i);

        //arcdiameter
        stars[i].d = (float)sq(dg*2.5e-4*gen_random_value_around(0.0, 0.25));
        stars[i].T = T;
        stars[i].colxyz = bb;
    }

    int diskW = diskHeight;
    int diskH = diskWidth;
    std::vector<float> disk(diskW*diskH, 0.0f);
    
    for(int l=2; l<=8; ++l)
    {
        auto f = pow(2.0, 1.0*l);
        auto rf = 1.0/f;
        for(int n=0; n<125*f; ++n)
        {
            auto y0 = (int)(sq(sq(gen_random_value_in_interval(0.0, 1.0)))*(diskH-15));
            auto h = (int)gen_random_value_in_interval(1, 2+f/150.);
            auto x0 = gen_random_value_in_interval(0, diskW);
            auto w = gen_random_value_in_interval(0.02*diskW, 0.1*diskW*(0.1+(double)y0/(1.0*diskH)));
            for(int yn=y0; yn<y0+h; ++yn)
            {
                
                auto x = (int)x0;
                for(int xn=0; xn<w; ++xn)
                {
                    if(x == diskW){ x = 0; }
                    disk[yn*diskW+x] += (float)(sq(f)*sq(y0/(1.0*diskH)));//(float)(f*y0*(diskH-y0)/(1.0f*sq(diskH)));
                    x += 1;
                }
            }
        }
    }

    auto diskImax = *std::max_element(disk.cbegin(), disk.cend());
    for(auto& x:disk){ x = cube(x/diskImax); }

    Params params;
    params.G = 1.0;
    params.M = 1.0;
    params.fovw = 75.0;
    params.zcam = 30.0;
    params.ycam = -2.00;
    params.T0disk = 3900.0;

    float Ifactor = 133.657e-6;
    float Iexp = 0.225f;
    float minI, maxI;

    bool trace_stop = false;

    TraceParams tpar;
    tpar.finish = &trace_stop;

    std::vector<Vec<float>> xyz_image;
    std::vector<sf::Color> result_image;

    sf::Image  img;
    sf::Texture tex;
    sf::Sprite sprite;
    
    auto resize_img = [&]
    {
        tpar.w = width;
        tpar.h = height;
        tpar.reset();
        xyz_image.resize(width*height);
        result_image.resize(width*height);
        for(int y=0; y<height; y+=1)
        {
            for(int x=0; x<width; x+=1)
            {
                xyz_image[y*width+x] = Vec<float>{0.f, 0.f, 0.f};
                result_image[y*width+x] = sf::Color::Black;
            }
        }
    };
    resize_img();

    auto update_img = [&]
    {
        if(!trace_stop)
        {
            trace_n(tpar, stars, starmap, xyz_image, result_image, disk, params);
        }
        
        minI = 1e19f;
        maxI = std::accumulate(xyz_image.cbegin(), xyz_image.cend(), 0.0f, [&](auto const& acc, auto const& col)
        {
            auto I = col.y;
            minI = I > 0 ? std::min(minI, I) : minI; return std::max(acc, I);
        });
        maxI *= Ifactor;
        minI *= 1.e1f;

        auto lmaxI = maxI;
        auto lminI = minI;
        auto range = lmaxI - lminI;
        for(int y=0; y<height; y+=1)
        {
            for(int x=0; x<width; x+=1)
            {
                auto xyz = xyz_image[y*width+x];   
                auto q = (float)pow(xyz.y/range, Iexp);
                if(q < 0.0f){ q = 0.0f; };
                auto rgb = convert_xyz_to_rgb(xyz);
                rgb = gamma_correct( redistribute_rgb( rgb / (1.0f) * q ), 2.4f );
                result_image[y*width+x] = rgb.sfColor();
            }
        }
       
        img.create(width, height, (sf::Uint8*)result_image.data());
        tex.loadFromImage(img);
        sprite.setTexture(tex, true);
        sprite.setPosition( sf::Vector2f(0, 0) );
    };

    auto draw_text = [&]
    {
        /*text.setString(std::string("Mass = ") + std::to_string(params.M));
        text.setPosition(30, 30);
        window.draw(text);

        text.setString(std::string("z = ") + std::to_string(params.zcam));
        text.setPosition(30, 50);
        window.draw(text);

        text.setString(std::string("y = ") + std::to_string(params.ycam));
        text.setPosition(30, 70);
        window.draw(text);

        text.setString(std::string("Imax = ") + std::to_string(maxI));
        text.setPosition(30, 90);
        window.draw(text);

        text.setString(std::string("Imin = ") + std::to_string(minI));
        text.setPosition(30, 110);
        window.draw(text);

        text.setString(std::string("Ifactor = ") + std::to_string(1e6*Ifactor));
        text.setPosition(30, 130);
        window.draw(text);

        text.setString(std::string("Iexp = ") + std::to_string(Iexp));
        text.setPosition(30, 150);
        window.draw(text);*/
    };

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
			if     (event.type == sf::Event::Closed  ){ window.close(); }
            //else if(event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Key::Left) { params.zcam -= 0.1; resize_img(); }
            //else if(event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Key::Right){ params.zcam += 0.1; resize_img(); }
            else if(event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Key::Up)   { Ifactor *= 1.5;/*params.ycam += 0.1; resize_img();*/ }
            else if(event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Key::Down) { Ifactor /= 1.5;/*params.ycam -= 0.1; resize_img();*/ }
            else if(event.type == sf::Event::KeyReleased && event.key.code == sf::Keyboard::Key::Tab) { trace_stop = !trace_stop; }
            else if(event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Key::Subtract){ Iexp -= 0.005; /*params.M -= 0.05; resize_img();*/ }
            else if(event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Key::Add)     { Iexp += 0.005; /*params.M += 0.05; resize_img();*/ }
			else if(event.type == sf::Event::Resized )
			{
                width  = event.size.width;
                height = event.size.height;
				window.setView(sf::View(sf::FloatRect(0.0f, 0.0f, width*1.0f, height*1.0f)));
				glViewport(0, 0, width, height);
				rect.setSize( sf::Vector2f(width*1.0f, height*1.0f) );
                resize_img();
			}
        }

        update_img();

        window.clear();
        window.draw(rect);
		window.draw(sprite);
        draw_text();
        window.display();
    }

    return 0;
}