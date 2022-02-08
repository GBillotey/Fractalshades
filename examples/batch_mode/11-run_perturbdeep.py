# -*- coding: utf-8 -*-
"""
===========================================
Ultra-deep embedded Julia set
===========================================

This example shows the kind of structure that occur very deep in the Mandelbrot
set. The width of this image is only 2.e-2608 [#f2]_.
This is not only below the separation power of double, but the delta are also
way below the minimal magnitude that can be stored in a double
(around  1e-323). A ad-hoc dataype is used internally during the Series
approximation step.

The period of the central minibrot is 2703248 ; the use of Series
approximations allows to skip around 5 millions iterations.

As the running time for this script is more than 30 minutes, this image
has been pre-computed.


.. [#f2] **Credit:** Coordinates courtesy of Microfractal :
        <https://www.deviantart.com/microfractal/art/Mandelbrot-Deep-Julia-Morphing-22-Golden-Sphere-886123409>
"""

import os
import numpy as np

import fractalshades as fs
import fractalshades.settings as fssettings

import fractalshades.models as fsm
import fractalshades.colors as fscolors
from fractalshades.postproc import (
    Postproc_batch,
    Continuous_iter_pp,
    Raw_pp,
    DEM_normal_pp,
)
from fractalshades.colors.layers import (
    Color_layer,
    Bool_layer,
    Blinn_lighting,
    Normal_map_layer,
)


def plot(directory):
    """
    Example plot of a very deep location
    Credit for the coordinates :
https://www.deviantart.com/microfractal/art/Mandelbrot-Deep-Julia-Morphing-22-Golden-Sphere-886123409
    """
    
    precision = 3520
    nx = 2400
    x = '-1.9409989391128007782656638595713128206620929316331395903205283705275932149841553750079140508152501109445961064000387852149507811657094626324996392008081820445955741490587617909708619603737265548027769325647808985287741667276189821676033432683374240723052323372896622554689290278821522432095519048328761094875168059910075072612524746195696519482376711787954155676296696827707057348137590781477540653443160271404114741216279924299516050033371623738987930710049260335938454436747992050897445704854917586460267198917634232454874517524790905068408711299098852857223323363509317448492707948571935557902448516804312250656708860690680767226144394692148838449346680921087412029850014210409147937112323614271639154365986968749816836442985665512979922489943829925482859841402388822224364772960765860128299173467963835512792813373451933644130190266047607001031626499249499592567711348988794983423352102489653363614657987130851011066068082416311059571884201802812522326939248656260215898332770887339844184688424916821959905805787211079924762420560654209080231130357236288188593275206143270109163936044056855567309338390204460230556526667618113052517191169646813610992208066490740332700166077086244561644939752386971282938070707062898838928187674154565542324706485606883204149973662143729325062503353762046809254607154103878222668282005954040495000651634097511941293052468376780564225465557438420172736278899353415715205080501056910932380856513690069593717239355697113322999606963893343303065997244593517188694362601778555657829079220370979486386183376634551544169026880446433151630826730127399985709844229666877539084763034446297595098204169627029966553348731711298433915468877133916519870332995252770006087468201433091412692008675169426600509762262849033820684824479730400854046509072164630272105114166613615665383021053646289448207336461725630828678598527683609575006544933462912457658924436663804582292428129309162915840098216747977268766925226272677267826315722555021136934491464926926641085339160830952887601459585519624489323898936587933143756193630971066578717659019875144049965572880866540996031144922280813352065159362962936897218127976473669535727210317367178865163942427120257230318803642220091013441782124465936161868040076934432584798273802125003893761405910549636791922164569969871504895180875775512279622397659490539731258965222183682582044022842758452337516752189727551206382556078493830490372988205049395299138260871313038171904760429268109644267193074206275040851482988811238053209498575928806745490180665861235757156293268030156174736154214485511919238045324816790747039434094153238651378208655247035749519428374239948111490578363711926298127059816373882058600875440218265729937727712935557101248183859985480838214443248343204994169001603385068217409551664275124868238925925271002064990910751541295196946319404974130124223074815816387748372081603618046256402766723419509314015491326315372861880224396707850752490829513864536227468094212074909783507683557390914984737208904927522859784984066452431380596052384391155762961147112917902257288838205513568126100751182438074841839964967562205987620459771593676482435160564881907643374624834394770129519338651384779340621276744712596399177749754956987947612707663018919330037816063293842647052555147743226921275393227281792532802856285703297338604821969492356674112869979073125870095512233460880231177088317720580337642382172126187069216048936896730950168087435988621276438670059341103609929304930466412268150569753470717829497601938341623581803667066999928999945000062'
    y = '-0.0006521165369165588520106289441620153907907521525225557951700039268755659160275378414816331241993503713942651869474366440330624054932785747734116130598457275168672169867853790149073948820621927863546898987531675745541556010963860271946131945706089440068213570737152573434606181998626256475661137064241766615685133034114571184540746713081041577482152866404680905298142203271097108866125320734562827910017740404764291477614758081664091324083106696109319507742512146578699926177581123430550120851818916049981949393089874937840577370413575565615246397463453690404270526656455145637869566754373564864548747775061651693403960187403612827482714675143082173905414385810506804378880397100996175280822311114495867725750471436402145707242763362689139153766093202506743259707579782531683072699910204376229255257696447791057044885184061849070063540925613028401048182129422816270970456315092465855569329878796473503666036123284601909076758201573065328180211040459230345709044071756847669905912521106047214804555579992552727318466143562534207465701332898411609149336015158023746864705973770293526683875460324480616782478489019514943512702395590818455582259983339029054638765126731537575594335734368117123722683120375030995584809981966023016675121788001130361752945926045051983789243281329028107416493849599211739205918880442308088915329310667744587253842928202077978689211781621700292204988439971992046135099101850443216579189710924423016693808479474589682525790322932538431715348758724089186172736870724706725359784401019519888555644853285575115223472590818823322033130852641478536530503881747200363162574382337579455223211205019832848615171631087121056343365803496414693646695845027511119821045191586941544022389773784151557473277272394880876628653639136977979073123486169650096416150642999247909147333278062324113459547152270378118487801961875006181455991513879900323624590458328414797373565255061007383050772917420374420930369627261609756033085579925058681478773760867701230719359928389502388023578804808713069253869301107296738982313988108484002367456921622985540672687977893371677916030176767500564905285025226973308704535270965189005321129735333599100313629076978281635241128387571784303118677495016595486491171040002394480779899042204488631259847989603182340726213078367178896618081990169319498713349339065257474424401748553283927933449943175175157120972516636257833849555669271463331231601029167028638597915809746995436188809835668111701784052366810307436108276491541042658178481843136392746657892940367221519240125914939061964441432380740020708127640600546604568699045234845708728863090863984386209155813013615576381026653379878624402126265227089167061378994809588030662831377110537145242600584959148498586439529663105983709419546957848439948376427305067215182145348517650481959560955434577158090652441197554228656503253796471623707876797570793456353888545895776536724341010890647565137237971578364800606022054805371016117249815862385204930532791360055457643453800167233033393824944921504096748637258867979270585206447548364249344195079436376739232814985700753366335710763351616828921383429188346008648781525793755795069682228036514982477038907976343304196109685257025904974333612600761354191140826329760186432247441069680365217200145218033541210372615053282512008534408785235009976598833958899392833195540809260984815364215770028371283427130718815533338521166040923413722562752702386025562655776477893889452984598715385588865771230862335806477085969230662862126372402082027768431991530300520064005268033000000000000000000'
    dx = '2.e-2608'

    # Computing full precision path with max_iter 2703248
    # SA stop 5321568

    test_corner = False
    if test_corner:
        import mpmath
        mpmath.mp.dps = precision
        x_corner = mpmath.mpf(x) + 0.5 * mpmath.mpf(dx)
        y_corner = mpmath.mpf(y) + 0.5 * mpmath.mpf(dx)
        x = str(x_corner)
        y = str(y_corner)


    # Set to True if you only want to rerun the post-processing part
    # even withg partially computed tiles
    fssettings.skip_calc = False
    # Set to True to enable multi-processing
    fssettings.enable_multithreading = True
    # Set to True to skip Newton iterations
    fssettings.no_newton = False
    # Set to True to enable additionnal text output
    fs.settings.inspect_calc = True

    calc_name="deep"
    colormap = fscolors.cmap_register["atoll"]

    f = fsm.Perturbation_mandelbrot(directory)
    f.zoom(
            precision=precision,
            x=x,
            y=y,
            dx=dx,
            nx=nx,
            xy_ratio=16./9.,
            theta_deg=0., 
            projection="cartesian",
            antialiasing=True)

    f.calc_std_div(
            datatype=np.complex128,
            calc_name=calc_name,
            subset=None,
            max_iter=10100100,
            M_divergence=1.e3,
            epsilon_stationnary=1.e-3,
            SA_params={
                "cutdeg": 64,
                "err": 1.e-6
            },
            interior_detect=False)

    f.run()

    # Plot the image
    pp = Postproc_batch(f, calc_name)
    pp.add_postproc("cont_iter", Continuous_iter_pp())
    pp.add_postproc("interior", Raw_pp("stop_reason", func="x != 1."))
    pp.add_postproc("DEM_map", DEM_normal_pp(kind="potential"))

    plotter = fs.Fractal_plotter(pp)   
    plotter.add_layer(Bool_layer("interior", output=False))
    plotter.add_layer(Normal_map_layer("DEM_map", max_slope=60, output=True))
    plotter.add_layer(Color_layer(
            "cont_iter",
            func="x",
            colormap=colormap,
            probes_z=[0.35, 0.9],
            probes_kind="relative",
            output=True
    ))

    plotter["cont_iter"].set_mask(
            plotter["interior"],
            mask_color=(0., 0., 0.)
    )
    plotter["DEM_map"].set_mask(
            plotter["interior"],
            mask_color=(0., 0., 0.)
    )

    # This is where we define the lighting (here 2 ccolored light sources)
    # and apply the shading
    light = Blinn_lighting(0.3, np.array([1., 1., 1.]))
    light.add_light_source(
        k_diffuse=0.2,
        k_specular=200.,
        shininess=400.,
        angles=(90., 20.),
        coords=None,
        color=np.array([0.5, 0.5, .4]))
    light.add_light_source(
        k_diffuse=1.5,
        k_specular=0.,
        shininess=0.,
        angles=(90., 40.),
        coords=None,
        color=np.array([1.0, 1.0, 1.0]))
    plotter["cont_iter"].shade(plotter["DEM_map"], light)
    

    plotter.plot()

def _plot_from_data(plot_dir):
    # Private function only used when building fractalshades documentation
    # This example takes too long too run to autogenerate the image for the
    # gallery each - so just grabbing the file from the html doc static path
    import PIL

    data_path = fs.settings.output_context["doc_data_dir"]
    im = PIL.Image.open(os.path.join(data_path, "gaia.jpg"))
    rgb_im = im.convert('RGB')
    tag_dict = {"Software": "fractalshades " + fs.__version__,
                "example_plot": "gaia"}
    pnginfo = PIL.PngImagePlugin.PngInfo()
    for k, v in tag_dict.items():
        pnginfo.add_text(k, str(v))
    if fs.settings.output_context["doc"]:
        fs.settings.add_figure(fs._Pillow_figure(rgb_im, pnginfo))
    else:
        # Should not happen
        raise RuntimeError()


if __name__ == "__main__":
    # Some magic to get the directory for plotting: with a name that matches
    # the file or a temporary dir if we are building the documentation
    try:
        realpath = os.path.realpath(__file__)
        plot_dir = os.path.splitext(realpath)[0]
        plot(plot_dir)
    except NameError:
        import tempfile
        with tempfile.TemporaryDirectory() as plot_dir:
            _plot_from_data(plot_dir)

