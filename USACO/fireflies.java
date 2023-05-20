import java.io.*;
import java.util.*;
import java.lang.*;
public class fireflies
{
   static boolean[] pressed = new boolean[4];
   static Set<String> set = new TreeSet<>();
   static ArrayList<Integer> on = new ArrayList<>();
   static ArrayList<Integer> off = new ArrayList<>();
   static int n, c;
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("fireflies.in"));
      n = in.nextInt();
      c = in.nextInt();
      int temp = in.nextInt();
      while(temp != -1)
      {
         on.add(temp - 1);
         temp = in.nextInt();
      }
      temp = in.nextInt();
      while(temp != -1)
      {
         off.add(temp - 1);
         temp = in.nextInt();
      }
      
      if(c > 3) c = 3;
      solve(0, c);
      
      for(String s : set) System.out.println(s);
      if(set.size() == 0) System.out.println("IMPOSSIBLE");
   }
   
   public static void solve(int pos, int pressLeft)
   {
      if(pos >= 4 || pressLeft <= 0)
      {
         boolean[] lights = new boolean[n];
         for(int i = 0; i < 4; i++)
            if(i == 0 && pressed[i])
               for(int j = 0; j < n; j++) lights[j] = !lights[j];
            else if(i == 1 && pressed[i])
               for(int j = 0; j < n; j += 2) lights[j] = !lights[j];
            else if(i == 2 && pressed[i])
               for(int j = 1; j < n; j += 2) lights[j] = !lights[j];
            else if(i == 3 && pressed[i])
               for(int j = 0; j < n; j += 3) lights[j] = !lights[j];
         
         for(int i = 0; i < on.size(); i++)
            if(lights[on.get(i)]) return;
         for(int i = 0; i < off.size(); i++)
            if(!lights[off.get(i)]) return;
         
         String s = "";
         for(boolean b : lights)
            if(b) s += "0";
            else s += "1";
         set.add(s);
      }
      
      if(pressLeft <= 0 || pos >= 4) 
         return;
      
      pressed[pos] = true;
      solve(pos + 1, pressLeft - 1);
      pressed[pos] = false;
      solve(pos + 1, pressLeft);
   }
}