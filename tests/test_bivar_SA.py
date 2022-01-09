# -*- coding: utf-8 -*-
import os
import unittest
import shutil

import numpy as np
import numba

import fractalshades as fs
import fractalshades.utils as fsutils
import fractalshades.colors as fscolors
import fractalshades.models as fsm
import fractalshades.numpy_utils.xrange as fsx
import fractalshades.numpy_utils.numba_xr as fsxn
# import fractalshades.bivar_series

from fractalshades.postproc import (
    Postproc_batch,
    Raw_pp,
    Attr_normal_pp,
    Attr_pp,
    Fractal_array
)
import test_config
from fractalshades.colors.layers import (
    Color_layer,
    Bool_layer,
    Normal_map_layer,
    Blinn_lighting
)

@numba.njit
def numba_path_do_nothing(path):
    # print("#######", path.has_xr)
    return path

@numba.njit
def numba_path_loop(path):
    npts = path.ref_path.size
    prev_idx = numba.int64(0)
    curr_xr = numba.int64(0)
    print("npts", npts)
    count = 0
    for j in range(npts):
        i = npts - j - 1
        (val, xr_val, is_xr, prev_idx, curr_xr
        ) = path.get(i, prev_idx, curr_xr)
        if is_xr:
            print("is_xr", i, val, xr_val)
            count += 1
        (val, xr_val, is_xr, prev_idx, curr_xr
        ) = path.get(i, prev_idx, curr_xr)
        if is_xr:
            print("is_xr", i, val, xr_val)
            count += 1
        i = j
        (val, xr_val, is_xr, prev_idx, curr_xr
        ) = path.get(i, prev_idx, curr_xr)
        if is_xr:
            print("is_xr", i, val, xr_val)
            count += 1
    print("count xr", count)
    assert count == 14 * 3

    for i in (7433792, 7433795, 8785472, 8785473, 9461312, 675720, 675728):
        (val, xr_val, is_xr, prev_idx, curr_xr
        ) = path.get(i, prev_idx, curr_xr)
        if is_xr:
            print("** is_xr", i, val, xr_val)
        else:
            print("** NOT xr", i, val, xr_val)

@numba.njit
def numba_c_from_pix(path, pix):
    return path.c_from_pix(pix)
    
    
class Test_ref_path(unittest.TestCase):
    
    @classmethod
    # @test_config.no_stdout
    def setUpClass(cls):
        fs.settings.enable_multiprocessing = True
        fs.settings.no_newton = True
        fs.settings.inspect_calc = True
        
        precision = 3520
        nx = 800
        x = '-1.9409989391128007782656638595713128206620929316331395903205283705275932149841553750079140508152501109445961064000387852149507811657094626324996392008081820445955741490587617909708619603737265548027769325647808985287741667276189821676033432683374240723052323372896622554689290278821522432095519048328761094875168059910075072612524746195696519482376711787954155676296696827707057348137590781477540653443160271404114741216279924299516050033371623738987930710049260335938454436747992050897445704854917586460267198917634232454874517524790905068408711299098852857223323363509317448492707948571935557902448516804312250656708860690680767226144394692148838449346680921087412029850014210409147937112323614271639154365986968749816836442985665512979922489943829925482859841402388822224364772960765860128299173467963835512792813373451933644130190266047607001031626499249499592567711348988794983423352102489653363614657987130851011066068082416311059571884201802812522326939248656260215898332770887339844184688424916821959905805787211079924762420560654209080231130357236288188593275206143270109163936044056855567309338390204460230556526667618113052517191169646813610992208066490740332700166077086244561644939752386971282938070707062898838928187674154565542324706485606883204149973662143729325062503353762046809254607154103878222668282005954040495000651634097511941293052468376780564225465557438420172736278899353415715205080501056910932380856513690069593717239355697113322999606963893343303065997244593517188694362601778555657829079220370979486386183376634551544169026880446433151630826730127399985709844229666877539084763034446297595098204169627029966553348731711298433915468877133916519870332995252770006087468201433091412692008675169426600509762262849033820684824479730400854046509072164630272105114166613615665383021053646289448207336461725630828678598527683609575006544933462912457658924436663804582292428129309162915840098216747977268766925226272677267826315722555021136934491464926926641085339160830952887601459585519624489323898936587933143756193630971066578717659019875144049965572880866540996031144922280813352065159362962936897218127976473669535727210317367178865163942427120257230318803642220091013441782124465936161868040076934432584798273802125003893761405910549636791922164569969871504895180875775512279622397659490539731258965222183682582044022842758452337516752189727551206382556078493830490372988205049395299138260871313038171904760429268109644267193074206275040851482988811238053209498575928806745490180665861235757156293268030156174736154214485511919238045324816790747039434094153238651378208655247035749519428374239948111490578363711926298127059816373882058600875440218265729937727712935557101248183859985480838214443248343204994169001603385068217409551664275124868238925925271002064990910751541295196946319404974130124223074815816387748372081603618046256402766723419509314015491326315372861880224396707850752490829513864536227468094212074909783507683557390914984737208904927522859784984066452431380596052384391155762961147112917902257288838205513568126100751182438074841839964967562205987620459771593676482435160564881907643374624834394770129519338651384779340621276744712596399177749754956987947612707663018919330037816063293842647052555147743226921275393227281792532802856285703297338604821969492356674112869979073125870095512233460880231177088317720580337642382172126187069216048936896730950168087435988621276438670059341103609929304930466412268150569753470717829497601938341623581803667066999928999945000062'
        y = '-0.0006521165369165588520106289441620153907907521525225557951700039268755659160275378414816331241993503713942651869474366440330624054932785747734116130598457275168672169867853790149073948820621927863546898987531675745541556010963860271946131945706089440068213570737152573434606181998626256475661137064241766615685133034114571184540746713081041577482152866404680905298142203271097108866125320734562827910017740404764291477614758081664091324083106696109319507742512146578699926177581123430550120851818916049981949393089874937840577370413575565615246397463453690404270526656455145637869566754373564864548747775061651693403960187403612827482714675143082173905414385810506804378880397100996175280822311114495867725750471436402145707242763362689139153766093202506743259707579782531683072699910204376229255257696447791057044885184061849070063540925613028401048182129422816270970456315092465855569329878796473503666036123284601909076758201573065328180211040459230345709044071756847669905912521106047214804555579992552727318466143562534207465701332898411609149336015158023746864705973770293526683875460324480616782478489019514943512702395590818455582259983339029054638765126731537575594335734368117123722683120375030995584809981966023016675121788001130361752945926045051983789243281329028107416493849599211739205918880442308088915329310667744587253842928202077978689211781621700292204988439971992046135099101850443216579189710924423016693808479474589682525790322932538431715348758724089186172736870724706725359784401019519888555644853285575115223472590818823322033130852641478536530503881747200363162574382337579455223211205019832848615171631087121056343365803496414693646695845027511119821045191586941544022389773784151557473277272394880876628653639136977979073123486169650096416150642999247909147333278062324113459547152270378118487801961875006181455991513879900323624590458328414797373565255061007383050772917420374420930369627261609756033085579925058681478773760867701230719359928389502388023578804808713069253869301107296738982313988108484002367456921622985540672687977893371677916030176767500564905285025226973308704535270965189005321129735333599100313629076978281635241128387571784303118677495016595486491171040002394480779899042204488631259847989603182340726213078367178896618081990169319498713349339065257474424401748553283927933449943175175157120972516636257833849555669271463331231601029167028638597915809746995436188809835668111701784052366810307436108276491541042658178481843136392746657892940367221519240125914939061964441432380740020708127640600546604568699045234845708728863090863984386209155813013615576381026653379878624402126265227089167061378994809588030662831377110537145242600584959148498586439529663105983709419546957848439948376427305067215182145348517650481959560955434577158090652441197554228656503253796471623707876797570793456353888545895776536724341010890647565137237971578364800606022054805371016117249815862385204930532791360055457643453800167233033393824944921504096748637258867979270585206447548364249344195079436376739232814985700753366335710763351616828921383429188346008648781525793755795069682228036514982477038907976343304196109685257025904974333612600761354191140826329760186432247441069680365217200145218033541210372615053282512008534408785235009976598833958899392833195540809260984815364215770028371283427130718815533338521166040923413722562752702386025562655776477893889452984598715385588865771230862335806477085969230662862126372402082027768431991530300520064005268033000000000000000000'
        dx = '2.e-2608'
        complex_type = np.complex128

        subset_dir = os.path.join(
            test_config.temporary_data_dir,
            "_numba_ref_path_dir"
        )
        fsutils.mkdir_p(subset_dir)
        cls.subset_dir = subset_dir
        cls.calc_name = "test"
        # cls.dir_ref = os.path.join(test_config.ref_data_dir, "subset_REF")
        cls.f = f = fsm.Perturbation_mandelbrot(subset_dir)

        f.zoom(precision=precision, x=x, y=y, dx=dx, nx=nx, xy_ratio=1.0,
               theta_deg=0., projection="cartesian", antialiasing=False)
        f.calc_std_div(
                datatype=complex_type,
                calc_name=cls.calc_name,
                subset=None,
                max_iter=10000000,
                M_divergence=1.e3,
                epsilon_stationnary=1.e-3,
                interior_detect=False,
                SA_params={"cutdeg": 2, "eps": 1.e-8},  # 7886 :  for 7884 partial
                calc_dzndc=False)
        print("f.iref", f.iref)
        f.iref = 0

        print("################# before get_FP_orbit")
        f.get_FP_orbit()
        print("################# after get_FP_orbit")
#        cls.FP_params = f.FP_params
#        cls.ref_path = f.ref_path


    def test_numba_path(self):
        """
        14 xr indices
        [ 675728 1351568 2027296 2703248 3378976 4054816 4730544 5406496 6082224
        6758064 7433792 8109744 8785472 9461312]"""
        
        ref_path = self.f.get_Ref_path()

        print("ref_path", ref_path)
        
        
#        ref_path.has_xr = False
#        print(ref_path.has_xr)
        new_path = numba_path_do_nothing(ref_path)
        print("new_path", new_path, new_path.ref_xr, new_path.ref_index_xr)
#        print(new_path.has_xr)
        numba_path_loop(ref_path)
#        print("~~~~~~~~~~~~~~~~")
#        print(self.FP_params["partials"].keys())
        
        print(new_path.dx)
        c = numba_c_from_pix(new_path, 0.5 + 0.5j)
        print(c)
        
    def test_print(self):
        fs.perturbation.PerturbationFractal.print_FP(
            self.FP_params, self.ref_path
        )
        
    
        
        


class Test_bivar_SA(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        fs.settings.enable_multiprocessing = True
        
        x = "-1.99996619445037030418434688506350579675531241540724851511761922944801584242342684381376129778868913812287046406560949864353810575744772166485672496092803920095332"
        y = "0.00000000000000000000000000000000030013824367909383240724973039775924987346831190773335270174257280120474975614823581185647299288414075519224186504978181625478529"
        dx = "1.8e-157"
        precision = 200
        nx = 1600
        complex_type = np.complex128

        subset_dir = os.path.join(
            test_config.temporary_data_dir,
            "_bivar_SA_dir"
        )
        fsutils.mkdir_p(subset_dir)
        cls.subset_dir = subset_dir
        cls.calc_name = "test"
        # cls.dir_ref = os.path.join(test_config.ref_data_dir, "subset_REF")
        cls.f = f = fsm.Perturbation_mandelbrot(subset_dir)

        f.zoom(precision=precision, x=x, y=y, dx=dx, nx=nx, xy_ratio=1.0,
               theta_deg=0., projection="cartesian", antialiasing=False)
        f.calc_std_div(
                datatype=complex_type,
                calc_name=cls.calc_name,
                subset=None,
                max_iter=100000,
                M_divergence=1.e3,
                epsilon_stationnary=1.e-3,
                interior_detect=False,
                SA_params={"cutdeg": 2, "eps": 1.e-8},  # 7886 :  for 7884 partial
                calc_dzndc=False)


        f.get_FP_orbit()
        Ref_path = f.get_Ref_path()

        # Initialise the Bivar_interpolator object
        kc = f.ref_point_kc()
        SA_params = f.SA_params
        SA_loop = f.SA_loop()
        bivar_interpolator = fsxn.make_Bivar_interpolator(
            Ref_path, SA_loop, kc, SA_params)

        # Jitted function used in numba inner-loop
        f._initialize = f.initialize()
        f._iterate = f.iterate()

        cls.bivar = bivar_interpolator



    # @test_config.no_stdout
    def test_basic(self):
        print("==============================")
        print("REF point orbit")
        for key, val in self.FP_params.items():
            print("==============================")
            print(key, ":")
            if key == "partials":
                for kp, vp in val.items():
                    print(kp, "-->", str(np.abs(vp)))
            else:
                print(val, ":")

#    # @test_config.no_stdout
#    def test_SA(self):
#        f = self.f
#        FP_params = self.FP_params
#        ref_path = self.ref_path
#        cutdeg = f.SA_params["cutdeg"]
#
#
#        kc = f.ref_point_scaling(f.iref, f.calc_name)
#        kc = fsx.mpf_to_Xrange(kc, dtype=f.base_float_type)
#        #kc = fsx.mpf_to_Xrange("1.e-120", dtype=f.base_float_type)
#        kc = kc.ravel() # Make it 1d for numba
#        kcX = np.insert(kc, 0, 0.)
#        kcX = fsx.Xrange_SA(kcX, cutdeg)
#        print("kc", kc, kcX)
#        
#        SA_loop = f.SA_loop()
#        P0 = fsx.Xrange_SA([0j], cutdeg=cutdeg)
#        n_iter = 0
#        SA_err_sq = 1.e-12
#        SA_stop = f.SA_params["SA_stop"]
#        ref_index_xr, ref_xr = f.get_ref_path_xr(FP_params)
#
#        ref_div_iter = 100000
#        
#        Pn, n, err = fsm.perturbation_mandelbrot.SA_run(
#            SA_loop, P0, n_iter, ref_path, kcX, SA_err_sq, SA_stop,
#            ref_index_xr, ref_xr, ref_div_iter
#        )
#        print("Pn:\n", Pn, n, err.view(fsx.Xrange_array))


    def test_bivar_SA(self):
        print("bvsa")
        bvi = self.bivar
        print("bvi", bvi)
        
        bi_attr_list = (
            "Ref_path",
#            "SA_loop",
            "min_seed_exp",
            "max_seed_exp",
            "bivar_SA_cutdeg",
            "bivar_SA_kc",
            "bivar_SA_eps",
#            "bivar_SA_lock", 
#            "bivar_SA_sto",
#            "bivar_SA_coeffs", 
#            "bivar_SA_computed", 
#            "bivar_SA_sq_zrad",
        )
        for attr in bi_attr_list:
            print(attr, getattr(bvi, attr))
        
        bvi_new = numba_box_bivar_interpolator(bvi)
        print("bvi_new", bvi_new)
        
        
        
        
@numba.njit
def numba_box_bivar_interpolator(bvi):
    # print(bvi)
    return bvi
        


if __name__ == "__main__":
    full_test = False
    runner = unittest.TextTestRunner(verbosity=2)
    if full_test:
        runner.run(test_config.suite([Test_bivar_SA]))
    else:
        suite = unittest.TestSuite()
#        suite.addTest(Test_bivar_SA("test_basic"))
##        suite.addTest(Test_bivar_SA("test_SA"))
#        suite.addTest(Test_bivar_SA("test_bivar_SA"))
        suite.addTest(Test_ref_path("test_numba_path"))
#        suite.addTest(Test_ref_path("test_print"))
        suite.addTest(Test_bivar_SA("test_bivar_SA"))
        runner.run(suite)
