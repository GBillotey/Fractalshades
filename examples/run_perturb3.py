# -*- coding: utf-8 -*-
import numpy as np

import fractalshades as fs
import fractalshades.models as fsm
import fractalshades.settings as settings
import fractalshades.colors as fscolors

def plot():
    """
    Example plot of "Dinkydau flake" location, classic test case for 
    perturbation techinque and glitch correction.
    """
    directory = "./perturb3"
    # A simple showcas using perturbation technique
    x, y = "-0.75", "0."
    dx = "5.e0"
    precision = 150
    nx = 3200
    x = '-1.993101465346811633595616349779370960719700854263460744227885419412572351324445321874004442403503011725492641453441484329872421586639050539267910275781311721259641283288898121495962960511188360859446141741747448336162741635241659807342073817485606900204314068415477260531866235822220430486119843399542682686903115170284286744427789259769672374264750048282753697939441835784223761880144973743249562058785490789121881765822494487680713802365108655723804325265573559505557675274602687698535315326824126504568612493712586162172913902182849175957355201038749736221172166381630971780664574945186600702295814202276821096082583371646391752258136082934974808859285874633438821365019578751567557825904520349615083013659980914914027970434909527071583051834592117828848476162531653958895351112086988431145593103631584906268842017812275327407852982198464690863881669210429524886644612378383194829443312845510612712652609161824242961337428831821452985753354390486290596759804822656081435972288149493458837417345622327424314356121019642859410599076344584439053'
    y = '-0.0000000000000000000001023710512431589570005203798273240286470771650981763139330824251367527539704908652681677255846096018123604758184274291544039306788236186699886780634842327075201516856958710101981983627439650646801970216473720458108184168523830340517163870169919249638047451318365933557979031258177237717753794769207546631778925762244933996868229775839453423976148160460368950010744615992543567659336441278697767936754177402161193387335896316048958999044786173769537636937690325372184302039387809032377492478883386867909568081378343026465718279702082155350634045451662043279571130329807181205168575035218579446441743895034500494162054542465662499561872308796389299543485971550119965555406797513540955180015002773089512902159298036751555750537198269153512054314778106379999904973319311775913118876664658896447125394230182911598486994660593785101070940474199338173235671779027464278763584882649334715308568951689930703319182855450455829071681801598812078546949506703052822064075723064430117087174494686425795962484572598182905448268661576794867514394048126125843131'
    dx = '0.000000000000000000000000000000000000000000000000000001611311677165311992036964188953416400357306950103215815939935672793028078345666262795353854791128193417180028322502563016371152356567374462535929127041725333256467561733290384009069197316210881426462964928434585521650089769009585599250595741136222346428026621232359069818161715344902647536565294138377179149230247747809792678090166161270842118179873716803137049807896171897484336423552754428273143300120233210420317384023816622150052145902931298541283952210479589483643604612244327904646956347253652556190458708451251053597744280720211750536552494355053069572018017158283906744473170534917744052363527030185810701281751448157641033852978421572439410367749824884649311769407121646490987840830692921695850309271975840408355787675238627661695949205488539553336065242972661368264433343340961667491522236015169240810915606280026794402643082165872947330057146892300382601081535552273108633947461317572323488134065825603499355505743581891071053495091972381402991134813796575186487618387985500615976611029133681690931704603523636857668558756510416666667'
    

    # Set to True if you only want to rerun the post-processing part
    settings.skip_calc = False
    # Set to True to enable multi-processing
    settings.enable_multiprocessing = True

#    xy_ratio = 1.0
#    theta_deg = 0.
    # complex_type = np.complex128

    mandelbrot = fsm.Perturbation_mandelbrot(directory)
    mandelbrot.zoom(
            precision=precision,
            x=x,
            y=y,
            dx=dx,
            nx=nx,
            xy_ratio=1.0,
            theta_deg=0., 
            projection="cartesian",
            antialiasing=True)

    mandelbrot.calc_std_div(
            complex_type=np.complex128,
            file_prefix="dev",
            subset=None,
            max_iter=50000,
            M_divergence=1.e3,
            epsilon_stationnary=1.e-3,
            pc_threshold=0.1,
            SA_params={"cutdeg": 64,
                       "cutdeg_glitch": 8,
                       "SA_err": 1.e-4,
                       "use_Taylor_shift": True},
            glitch_eps=1.e-6,
            interior_detect=False, #True,
            glitch_max_attempt=20)

    mandelbrot.run()


    mask_codes = [0, 2, 3, 4]
    mask = fs.Fractal_Data_array(mandelbrot, file_prefix="dev",
        postproc_keys=('stop_reason', lambda x: np.isin(x, mask_codes)),
        mode="r+raw")
#    cv = fs.Fractal_Data_array(mandelbrot, file_prefix="dev",
#                postproc_keys=('stop_reason', lambda x: x != 2), mode="r+raw")
#    potential_data_key = ("potential", {})


#    gold = np.array([255, 210, 66]) / 255.
#    black = np.array([0, 0, 0]) / 255.
#    color_gradient = fs.Color_tools.Lch_gradient(gold, black, 200)
#    colormap = fs.Fractal_colormap(color_gradient)
    gold = np.array([255, 210, 66]) / 255.
    black = np.array([0, 0, 0]) / 255.
    purple = np.array([181, 40, 99]) / 255.
    colors1 = np.vstack((purple[np.newaxis, :]))
    colors2 = np.vstack((gold[np.newaxis, :]))
    colormap = fscolors.Fractal_colormap(kinds="Lch", colors1=colors1,
            colors2=colors2, n=200, funcs=None, extent="mirror")

    plotter = fs.Fractal_plotter(
        fractal=mandelbrot,
        base_data_key=("potential", {}), # potential_data_key, #glitch_sort_key,
        base_data_prefix="dev",
        base_data_function=lambda x:x,# np.sin(x*0.0001),
        colormap=colormap,
        probes_val=[0., 0.55],# 200. + 200, #* 428  - 00.,#[0., 0.5, 1.], #phi * k * 2. + k * np.array([0., 1., 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]) / 3.5,
        probes_kind="qt",#"z", "qt"
        mask=mask)
    
    
    #plotter.add_calculation_layer(postproc_key=potential_data_key)
    
#    layer1_key = ("DEM_shade", {"kind": "potential",
#                                "theta_LS": 30.,
#                                "phi_LS": 50.,
#                                "shininess": 3.,
#                                "ratio_specular": 15000.})
#    plotter.add_grey_layer(postproc_key=layer1_key, intensity=0.75, 
#                         blur_ranges=[],#[[0.99, 0.999, 1.0]],
#                        disp_layer=False, #skewness=0.2,
#                         normalized=False, hardness=0.35,  
#            skewness=0.0, shade_type={"Lch": 1.0, "overlay": 1., "pegtop": 4.})
    
#    layer2_key = ("field_lines", {})
#    plotter.add_grey_layer(postproc_key=layer2_key,
#                         hardness=1.0, intensity=0.68, skewness=0.0,
##                         blur_ranges=[[0.50, 0.60, 1.0]], 
#                         shade_type={"Lch": 0., "overlay": 2., "pegtop": 1.}) 
    plotter.add_grey_layer(
                postproc_key=("DEM_shade", {"kind": "potential",
                                "theta_LS": 30.,
                                "phi_LS": 50.,
                                "shininess": 30.,
                                "ratio_specular": 15000.}),
                blur_ranges=[],
                hardness=0.9,
                intensity=0.8,
                shade_type={"Lch": 1.0, "overlay": 0., "pegtop": 4.})


    plotter.plot("dev", mask_color=(0., 0., 0.))

if __name__ == "__main__":
    plot()
